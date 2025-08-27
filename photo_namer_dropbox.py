import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from datetime import datetime
import threading
import math
import io
import dropbox
from dropbox.exceptions import ApiError
from dropbox.files import FileMetadata
import dropbox.common
from concurrent.futures import ThreadPoolExecutor


# ==============================================================================
# CORE IMAGE ANALYSIS LOGIC (MAXIMUM ACCURACY VERSION)
# ==============================================================================

def get_all_image_files_recursively(dbx, root_path):
    print(f"🔎 Починаю рекурсивне сканування папки: {root_path}")
    image_paths = []
    try:
        result = dbx.files_list_folder(root_path, recursive=True)
        while True:
            for entry in result.entries:
                if isinstance(entry, FileMetadata) and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(entry.path_display)
            if not result.has_more: break
            result = dbx.files_list_folder_continue(result.cursor)
    except ApiError as e:
        print(f"ПОМИЛКА API при скануванні папки {root_path}: {e}")
        return None
    print(f"✅ Сканування завершено. Знайдено {len(image_paths)} зображень.")
    return image_paths


def process_single_archive_image(args):
    """Завантажує та обробляє одне зображення за допомогою SIFT."""
    file_path, dbx, feature_detector = args
    try:
        _, res = dbx.files_download(file_path)
        img_bytes = res.content
        img_np = np.frombuffer(img_bytes, np.uint8)
        archive_img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        if archive_img is None: return None

        kps, des = feature_detector.detectAndCompute(archive_img, None)
        if des is None or len(kps) < 10: return None

        # SIFT дескриптори мають тип float32, це важливо для FLANN
        return {'file': os.path.basename(file_path), 'kps': kps, 'descriptors': des.astype(np.float32)}
    except Exception:
        return None


def create_feature_database(all_image_paths, dbx, feature_detector, status_callback):
    """Створює базу даних ознак SIFT, використовуючи багатопоточність."""
    feature_database = []
    total_files = len(all_image_paths)

    with ThreadPoolExecutor(max_workers=8) as executor:
        args_list = [(path, dbx, feature_detector) for path in all_image_paths]
        results = executor.map(process_single_archive_image, args_list)

        for i, result in enumerate(results):
            status_callback(f"Створення бази даних SIFT: {i + 1}/{total_files}...")
            if result:
                feature_database.append(result)

    return feature_database


def compare_single_entry(args):
    """
    Виконує порівняння для одного запису з бази даних.
    Ця функція буде запускатись у паралельних потоках.
    """
    target_kps, target_des, db_entry, flann = args
    archive_kps = db_entry['kps']
    archive_des = db_entry['descriptors']

    matches = flann.knnMatch(target_des.astype(np.float32), archive_des, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    num_inliers = 0
    if len(good_matches) > 10:
        src_pts = np.float32([target_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([archive_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            num_inliers = mask.ravel().sum()

    return {'file': db_entry['file'], 'inliers': num_inliers}

def find_best_match_accurate(target_kps, target_des, feature_database, status_callback):
    """
    Шукає найкращий збіг, використовуючи ThreadPoolExecutor для паралельного аналізу.
    """
    if target_des is None: return None

    # Налаштування для FLANN, оптимізованого для SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    best_match = {'file': None, 'inliers': -1}

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Створюємо список аргументів для кожного завдання
        args_list = [(target_kps, target_des, entry, flann) for entry in feature_database]

        results = executor.map(compare_single_entry, args_list)

        total_db_items = len(feature_database)
        for i, result in enumerate(results):
            status_callback(f"Порівняння з базою: {i + 1}/{total_db_items}...")
            if result and result['inliers'] > best_match['inliers']:
                best_match = result

    return best_match['file']

def analyze_image_data(image_bytes, feature_database, feature_detector, status_callback):
    img_np = np.frombuffer(image_bytes, np.uint8)
    img_with_frame_color = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    if img_with_frame_color is None: return "Error loading image", "Error"

    h, w = img_with_frame_color.shape[:2]
    common_divisor = math.gcd(h, w)
    aspect_ratio = f"{w // common_divisor}X{h // common_divisor}"

    img_with_frame = cv2.cvtColor(img_with_frame_color, cv2.COLOR_BGR2GRAY)
    border_y, border_x = int(h * 0.03), int(w * 0.03)  # Зменшуємо обрізку до 3%
    content_img = img_with_frame[border_y: h - border_y, border_x: w - border_x]

    content_kps, content_des = feature_detector.detectAndCompute(content_img, None)

    original_photo_file = find_best_match_accurate(content_kps, content_des, feature_database, status_callback)
    if original_photo_file is None: original_photo_file = "Photo Not Found"

    return original_photo_file, aspect_ratio


# ==============================================================================
# GRAPHICAL USER INTERFACE (GUI) - логіка залишається тією ж
# ==============================================================================

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.feature_detector = None
        self.dbx = None
        self.feature_database = []
        self.input_image_bytes = None
        self.title("Генератор назв для фото (High Accuracy)")
        self.geometry("800x600")
        ctk.set_appearance_mode("Dark")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        # ... (решта коду GUI залишається без змін) ...
        self.left_frame = ctk.CTkFrame(self, width=400, corner_radius=0)
        self.left_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.select_button = ctk.CTkButton(self.left_frame, text="Обрати фото", command=self.select_image)
        self.select_button.grid(row=0, column=0, padx=20, pady=20)
        self.image_label = ctk.CTkLabel(self.left_frame, text="Тут буде ваше фото", text_color="gray")
        self.image_label.grid(row=1, column=0, padx=20, pady=20)
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.right_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.right_frame, text="Шаблон:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.pliv_entry = ctk.CTkEntry(self.right_frame, placeholder_text="PLIV1")
        self.pliv_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkLabel(self.right_frame, text="Версія (V):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.version_entry = ctk.CTkEntry(self.right_frame, placeholder_text="V1")
        self.version_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkLabel(self.right_frame, text="Мова:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.lang_var = ctk.StringVar(value="CH")
        self.lang_menu = ctk.CTkOptionMenu(self.right_frame, values=["CH", "DE", "DK", "EN", "FI", "NO", "SE"],
                                           variable=self.lang_var)
        self.lang_menu.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkLabel(self.right_frame, text="DS:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.ds_var = ctk.StringVar(value="DS1")
        self.ds_menu = ctk.CTkOptionMenu(self.right_frame, values=["DS1", "DS2", "DS3"], variable=self.ds_var)
        self.ds_menu.grid(row=3, column=1, padx=10, pady=10, sticky="ew")
        self.generate_button = ctk.CTkButton(self, text="🚀 Генерувати назву", command=self.start_analysis_thread,
                                             state="disabled")
        self.generate_button.grid(row=1, column=1, padx=20, pady=10, sticky="ew")
        self.result_entry = ctk.CTkEntry(self, placeholder_text="Результат...", font=("Arial", 14))
        self.result_entry.grid(row=2, column=1, padx=20, pady=10, sticky="ew")
        self.status_label = ctk.CTkLabel(self, text="Запуск програми...")
        self.status_label.grid(row=3, column=1, padx=20, pady=10, sticky="ew")
        threading.Thread(target=self.initialize_app).start()

    def initialize_app(self):
        try:
            from config import DROPBOX_TOKEN
        except ImportError:
            self.update_status("Помилка: Створіть config.py з DROPBOX_TOKEN")
            return
        if not DROPBOX_TOKEN or "sl.B..." in DROPBOX_TOKEN:
            self.update_status("Помилка: Вставте токен у config.py")
            return
        try:
            self.update_status("Підключення до Dropbox...")
            dbx_base_client = dropbox.Dropbox(DROPBOX_TOKEN)
            account_info = dbx_base_client.users_get_current_account()
            root_info = account_info.root_info
            if isinstance(root_info, dropbox.common.TeamRootInfo):
                self.dbx = dbx_base_client.with_path_root(
                    dropbox.common.PathRoot.namespace_id(root_info.root_namespace_id))
            else:
                self.dbx = dbx_base_client
            self.update_status("Сканування файлів у Dropbox...")
            all_image_paths = get_all_image_files_recursively(self.dbx, ROOT_SEARCH_PATH_DBX)
            if all_image_paths is None:
                raise Exception("Шлях не знайдено, перевірте налаштування.")
            self.update_status("Ініціалізація детектора SIFT...")
            self.feature_detector = cv2.SIFT_create()  # Використовуємо SIFT
            self.feature_database = create_feature_database(all_image_paths, self.dbx, self.feature_detector,
                                                            self.update_status)
            self.generate_button.configure(state="normal")
            self.update_status("✅ Готовий до роботи!")
        except Exception as e:
            self.update_status(f"Помилка ініціалізації: {e}")

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not file_path: return
        with open(file_path, "rb") as f:
            self.input_image_bytes = f.read()
        img = Image.open(io.BytesIO(self.input_image_bytes))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(350, 350 * img.height / img.width))
        self.image_label.configure(image=ctk_img, text="")

    def start_analysis_thread(self):
        if self.input_image_bytes is None:
            self.update_status("Помилка: спочатку оберіть фото!")
            return
        self.generate_button.configure(state="disabled", text="Аналіз...")
        threading.Thread(target=self.analysis_worker).start()

    def update_status(self, text):
        self.after(0, self._update_status_safe, text)

    def _update_status_safe(self, text):
        self.status_label.configure(text=text)

    def analysis_worker(self):
        pliv = self.pliv_entry.get() or "PLIV1"
        version = self.version_entry.get() or "V1"
        lang = self.lang_var.get()
        ds = self.ds_var.get()
        found_photo, aspect_ratio = analyze_image_data(self.input_image_bytes, self.feature_database,
                                                       self.feature_detector, self.update_status)
        if "Not Found" in found_photo or "Error" in found_photo:
            final_name = f"ПОМИЛКА: {found_photo}"
        else:
            base_name = os.path.splitext(found_photo)[0]
            today_date = datetime.now().strftime('%Y%m%d')
            final_name = f"{base_name}_{pliv}_IMG{aspect_ratio}_{version}_{lang}_{ds}_{today_date}"
        self.after(0, self.update_ui_with_results, final_name)

    def update_ui_with_results(self, final_name):
        self.result_entry.delete(0, "end")
        self.result_entry.insert(0, final_name)
        self.status_label.configure(text="✅ Готово! Назву згенеровано.")
        self.generate_button.configure(state="normal", text="🚀 Генерувати назву")


# ==============================================================================
# Головний блок запуску програми
# ==============================================================================
if __name__ == '__main__':
    # --- ❗ ВАЖЛИВІ НАЛАШТУВАННЯ ---
    ROOT_SEARCH_PATH_DBX = "/TestAssets"
    # ---------------------------
    app = App()
    app.mainloop()