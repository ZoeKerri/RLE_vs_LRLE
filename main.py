import pygame
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
import cv2
from lossless_RLE import RLE
from RLE_LRLE_utilities import Utilities
from lossy_RLE import Lossy_RLE
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Khởi tạo pygame
pygame.init()

# Kích thước màn hình
screen = pygame.display.set_mode((1956, 840))

# Màu sắc
WHITE = (255, 255, 255)
GRAY = (220, 220, 220)
BLACK = (0, 0, 0)
HOVER_COLOR = (180, 180, 180)

# Biến để lưu trạng thái
uploaded_image = None
compressed_image_RLE = None
compressed_image_LRLE_dropbit = None
compressed_image_LRLE_blending = None

image_name = ""
image_full_path = ""
classification_result = "Chưa có kết quả"
selected_folder = None
image_files = []
current_image_index = 0
original_size = 0
lossless_compressed = None
lossy_compressed_dropbit = None
lossy_compressed_blending = None
is_rgb = False

# Khởi tạo RLE và Lossy_RLE
rle_instance = RLE()
lossy_rle = Lossy_RLE()

# Tải phông chữ
font_path = "NotoSans-Regular.ttf"
try:
    title_font = pygame.font.Font(font_path, 36)  # Tiêu đề chính
    font = pygame.font.Font(font_path, 16)        # Tiêu đề phụ, thông tin
    button_font = pygame.font.Font(font_path, 24) # Chữ nút
    info_font = pygame.font.Font(font_path, 16)    # Thông tin chi tiết
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy phông chữ '{font_path}'. Đảm bảo tệp tồn tại.")
    exit()

red_title_surf = title_font.render("So sánh", True, (255, 0, 0))
red_title_rect = red_title_surf.get_rect()
rest_title_surf = title_font.render(" nén RLE và RLE Lossy", True, BLACK)
rest_title_rect = rest_title_surf.get_rect()
total_width = red_title_rect.width + rest_title_rect.width
x_start = 750
y = 40 - (red_title_rect.height // 2)
red_title_rect.topleft = (x_start, y)
rest_title_rect.topleft = (x_start + red_title_rect.width, y)

def get_none():
    global uploaded_image, image_name, image_full_path, classification_result, original_size, lossless_compressed, lossy_compressed_dropbit, lossy_compressed_blending
    global compressed_image_LRLE_blending, compressed_image_LRLE_dropbit, compressed_image_RLE
    uploaded_image = None
    compressed_image_RLE = None
    compressed_image_LRLE_dropbit = None
    compressed_image_LRLE_blending = None

    image_name = ""
    image_full_path = ""
    classification_result = "Chưa có kết quả"
    original_size = 0
    lossless_compressed = None
    lossy_compressed_dropbit = None
    lossy_compressed_blending = None

# Hàm tải ảnh
def upload_image():
    global uploaded_image, image_name, image_full_path, classification_result, original_size, lossless_compressed, lossy_compressed_dropbit, is_rgb
    get_none()
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
    filetypes=[
        ("Image files", "*.bmp *.jpg *.jpeg *.png *.gif *.webp"),
        ("Bitmap files", "*.bmp"),
        ("JPEG files", "*.jpg *.jpeg"),
        ("PNG files", "*.png"),
        ("GIF files", "*.gif"),
        ("All files", "*.*")
        ]
    )
    if file_path:
        uploaded_image = pygame.image.load(file_path)
        uploaded_image = pygame.transform.scale(uploaded_image, (200, 220))
        bitsize = uploaded_image.get_bitsize()
        if bitsize == 24 or bitsize == 32:
            is_rgb = True
        else:
            is_rgb = False
        image_name = os.path.basename(file_path)
        image_full_path = file_path
        original_size = os.path.getsize(file_path)
        classification_result = "Chưa có kết quả"
        lossless_compressed = None
        lossy_compressed_dropbit = None
        lossy_compressed_blending = None
    else:
        get_none()
        

# Hàm tải thư mục
def upload_folder():
    global selected_folder, image_files, current_image_index, classification_result, image_name, original_size, lossless_compressed, lossy_compressed_dropbit, lossy_compressed_blending, compressed_image_RLE, is_rgb, image_full_path
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    if folder_path:
        selected_folder = folder_path
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png', '.gif', '.webp'))
        ]
        image_files = sorted(image_files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
        if image_files:
            current_image_index = 0
            load_image()
            classification_result = "Chưa có kết quả"
            lossless_compressed = None
            lossy_compressed_dropbit = None
            lossy_compressed_blending = None
            image_full_path = ""
    else:
            get_none()
            

def show_result_dialog_matplotlib(title, message, decompressed_img_path):
    """Hiển thị ảnh và thông tin bằng Matplotlib, hỗ trợ cả ảnh màu và trắng đen."""
    try:
        # Đọc ảnh bằng Matplotlib
        img = mpimg.imread(decompressed_img_path)

        # Tạo figure và axes
        fig, ax = plt.subplots()

        # Kiểm tra số chiều của ảnh để xác định là trắng đen hay ảnh màu
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')  # Ảnh trắng đen
        else:
            ax.imshow(img)  # Ảnh màu

        ax.axis('off')  # Tắt trục tọa độ

        # Tạo text để hiển thị thông tin
        info_text = message

        # Hiển thị text bên dưới ảnh
        plt.text(0.5, -0.15, info_text, ha='center', va='center', transform=ax.transAxes)
        plt.subplots_adjust(bottom=0.25)  # Điều chỉnh khoảng trống dưới ảnh

        plt.title(title)
        plt.show()

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh tại đường dẫn '{decompressed_img_path}'")
    except Exception as e:
        print(f"Lỗi hiển thị ảnh bằng Matplotlib: {e}")

def upload_encoded_file():
    file_path = filedialog.askopenfilename(
        title="Chọn file .npz đã mã hóa",
        filetypes=[("NPZ files", "*.npz")]
    )

    if not file_path:
        print("Lỗi đọc file!")
        return

    try:
        filename = os.path.basename(file_path)
        with np.load(file_path, allow_pickle=True) as data:
            compressed_data = data['compressed_data']
            original_shape = data['original_shape']
            method_name = data['method_name']
            
        size_bytes = compressed_data.nbytes
        size_kb = round(size_bytes / 1024, 2)

        size_on_disk = os.path.getsize(file_path)

        decompressed_img, shape = Utilities.load_and_decompress(file_path)
        if(len(shape) == 3):
            decompressed_img = cv2.cvtColor(decompressed_img, cv2.COLOR_RGB2BGR)

        output_path = f"Archive/DecodingFile/DECODING_{filename}.bmp"
        cv2.imwrite(output_path, decompressed_img)

        #lý do lấy kích thước của ảnh giải nén làm kích thước gốc là vì ảnh khi giải nén ra đều là cùng kích thước và kênh màu, không bị resize gì hết nên ta lấy làm kích thước gốc
        original_size_byte = os.path.getsize(output_path)

        message = (
            f"Tên file: {filename}\n"
            f"Kích thước mảng nén: {size_bytes} bytes (~{size_kb} KB)\n"
            f"Kích thước file thực tế lưu trên đĩa: {size_on_disk} bytes (~{size_on_disk/1024:.2f} KB)\n"
        )
        show_result_dialog_matplotlib("Kết quả", message, output_path) # Sử dụng hàm Matplotlib

    except Exception as e:
        print(f"Lỗi đọc file npz: {e}")

# Hàm tải ảnh hiện tại
def load_image():
    global uploaded_image, image_name, classification_result, original_size, lossless_compressed, lossy_compressed_dropbit, is_rgb, lossy_compressed_blending
    if image_files and selected_folder:
        file_path = os.path.join(selected_folder, image_files[current_image_index])
        try:
            uploaded_image = pygame.image.load(file_path)
            bitsize = uploaded_image.get_bitsize()

            if bitsize == 24 or bitsize == 32:
                is_rgb = True
            else:
                is_rgb = False

            uploaded_image = pygame.transform.scale(uploaded_image, (200, 220))
            image_name = image_files[current_image_index]
            original_size = os.path.getsize(file_path)
            classification_result = "Chưa có kết quả"
            lossless_compressed = None
            lossy_compressed_dropbit = None
            lossy_compressed_blending = None
        except pygame.error as e:
            print(f"Lỗi khi tải ảnh: {file_path} - {e}")
            uploaded_image = None
            image_name = ""
            original_size = 0
            classification_result = f"Lỗi tải ảnh: {os.path.basename(file_path)}"

def get_dtype_from_maxval(max_val):
    if max_val <= np.iinfo(np.uint8).max:  # 255
        return np.uint8
    elif max_val <= np.iinfo(np.uint16).max:  # 65535
        return np.uint16
    elif max_val <= np.iinfo(np.uint32).max:  # 4294967295
        return np.uint32
    else:
        return np.uint64

def calc_the_size_of_image(img_path):
    bgr_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if bgr_img is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {img_path}")

    # Kiểm tra số kênh của ảnh
    is_grayscale = len(bgr_img.shape) == 2 or bgr_img.shape[2] == 1

    # Flatten ảnh
    if is_grayscale:
        # Ảnh xám (1 kênh)
        if len(bgr_img.shape) == 3:
            img = bgr_img[:, :, 0]  # Lấy kênh đầu tiên nếu ảnh 3 chiều
        else:
            img = bgr_img
        flat_img = img.flatten()  # Làm phẳng thành mảng 1D
    else:
        # Ảnh màu (RGB)
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
        flat_img = img.reshape(-1)  # Làm phẳng thành [R1, G1, B1, R2, G2, B2, ...]

    # Tính kích thước
    dtype = np.uint8  # Ảnh thường là uint8 (1 byte mỗi pixel)
    flat_img_np = np.array(flat_img, dtype=dtype)
    size_bytes = flat_img_np.nbytes  # Tổng số byte
    size_kb = size_bytes / 1024  # Chuyển sang KB

    return size_kb

def decompress_and_save(compressed_data, original_shape, output_path, method_name="1"):
    """
    Giải nén dữ liệu nén và lưu thành ảnh.
    Args:
        compressed_data: Mảng dữ liệu đã nén (list hoặc numpy array).
        original_shape: Shape của ảnh gốc (tuple, ví dụ: (height, width, 3) hoặc (height, width)).
        output_path: Đường dẫn lưu ảnh đầu ra (ví dụ: 'output_image.png').
        method_name: Phương thức nén (mặc định là "1").
    Returns:
        None
    """
    # Khởi tạo đối tượng Utilities
    util = Utilities()

    # Giải nén dữ liệu
    decompressed_img = util.rle_lrle_decompressing(compressed_data, original_shape, method_name)

    # Kiểm tra dữ liệu giải nén
    if decompressed_img is None:
        print("Giải nén thất bại!")
        return

    # Nếu là ảnh màu (RGB), đảm bảo đúng thứ tự kênh cho OpenCV (BGR)
    if len(original_shape) == 3 and original_shape[2] == 3:
        decompressed_img = cv2.cvtColor(decompressed_img, cv2.COLOR_RGB2BGR)

    # Lưu ảnh
    cv2.imwrite(output_path, decompressed_img)
    print(f"Ảnh đã được lưu tại: {output_path}")


# Hàm chạy thuật toán và so sánh nén
def run_algorithm():
    global classification_result, lossless_compressed, lossy_compressed_dropbit, lossy_compressed_blending, compressed_image_RLE, compressed_image_LRLE_blending, compressed_image_LRLE_dropbit
    if not uploaded_image or not image_name:
        classification_result = "Chưa có hình ảnh để nén"
        return

    try:
        img_path = image_full_path if image_full_path else os.path.join(selected_folder, image_name)

        # Nén RLE lossless
        rle_instance = RLE("")
        lrle_instance = Lossy_RLE()
    
        #Nén RLE lossless
        #image_shape đc chia sẻ chung cho 2 thuật toán nên chỉ cần bên rle return về là được
        compress_arr_rle, img_shape_rle, max_val_rle = rle_instance.get_compressed_data(img_path)
        method_name_rle = rle_instance.best_method_name


        compress_arr_rle_np = np.array(compress_arr_rle, dtype = get_dtype_from_maxval(max_val_rle))

        #Lý do dùng np.uint32 là do bên lossy có thể tiến hành gộp hoặc cộng giá trị tạo nên giá trị có thể lớn hơn giá trị kiểu uint8 hoặc uint16
        compress_arr_rle_uint32 = compress_arr_rle_np.astype(np.uint32)
        compressed_image_LRLE_dropbit,  max_val_lrle_dropbit = lrle_instance.rle_lossy_compressing_dropbit(compress_arr_rle_uint32,method_name_rle,max_val_rle,drop_bits = 5, short_run_threshold = 3)
        compressed_image_LRLE_blending,  max_val_lrle_blending = lrle_instance.rle_lossy_blending_short_runs(compress_arr_rle_uint32,method_name_rle,max_val_rle,short_run_threshold = 3, blending_threshold = 5)

        # Tính kích thước cho mảng nén RLE
        lossless_compressed = compress_arr_rle_np 
        size_rle = compress_arr_rle_np.nbytes  
        size_kb_rle = size_rle / 1024  
        decompress_and_save(lossless_compressed, img_shape_rle, f"Archive/DecodingFile/RLE_{image_name}.bmp", method_name_rle)
        Utilities.export_compressed_data(lossless_compressed,img_shape_rle,method_name_rle, f"Archive/EncodingFIle/RLE_encoding_file_{image_name}.npz")

        # Tính kích thước cho mảng nén LRLE Dropbit
        lossy_compressed_dropbit = compressed_image_LRLE_dropbit
        size_lrle_dropbit = compressed_image_LRLE_dropbit.nbytes  
        size_kb_lrle_dropbit = size_lrle_dropbit / 1024
        #method_name_lrle chỉ dùng để in ra thôi chứ method_name_rle là dùng để xác định ảnh xám hay ảnh màu
        decompress_and_save(lossy_compressed_dropbit, img_shape_rle, f"Archive/DecodingFile/LRLE_dropbit_{image_name}.bmp", method_name_rle)
        Utilities.export_compressed_data(lossy_compressed_dropbit,img_shape_rle,method_name_rle, f"Archive/EncodingFIle/LRLE_dropbit_encoding_file_{image_name}.npz")

        # Tính kích thước cho mảng nén LRLE Blending
        lossy_compressed_blending = compressed_image_LRLE_blending
        size_lrle_blending = compressed_image_LRLE_blending.nbytes  
        size_kb_lrle_blending = size_lrle_blending / 1024
        #method_name_lrle chỉ dùng để in ra thôi chứ method_name_rle là dùng để xác định ảnh xám hay ảnh màu
        decompress_and_save(lossy_compressed_blending, img_shape_rle, f"Archive/DecodingFile/LRLE_blending_{image_name}.bmp", method_name_rle)
        Utilities.export_compressed_data(lossy_compressed_blending,img_shape_rle,method_name_rle, f"Archive/EncodingFIle/LRLE_blending_encoding_file_{image_name}.npz")

        #vẽ hình nén lên
        compressed_image_RLE = pygame.image.load(f"Archive/DecodingFile/RLE_{image_name}.bmp")
        compressed_image_RLE = pygame.transform.scale(compressed_image_RLE, (200, 220))

        compressed_image_LRLE_dropbit = pygame.image.load(f"Archive/DecodingFile/LRLE_dropbit_{image_name}.bmp")
        compressed_image_LRLE_dropbit = pygame.transform.scale(compressed_image_LRLE_dropbit, (200, 220))

        compressed_image_LRLE_blending = pygame.image.load(f"Archive/DecodingFile/LRLE_blending_{image_name}.bmp")
        compressed_image_LRLE_blending = pygame.transform.scale(compressed_image_LRLE_blending, (200, 220))

        screen.blit(uploaded_image, (1594, 220))

    except Exception as e:
        classification_result = f"Lỗi xử lý ảnh: {str(e)}"
        print(classification_result)

def shorten_text(file_name):
  if len(file_name) <= 50:
    return file_name
  else:
    prefix = file_name[:29]
    suffix = file_name[-20:]
    return f"{prefix}...{suffix}"


# Vòng lặp chính
running = True
while running:
    screen.fill(WHITE)

    # Tiêu đề chính
    screen.blit(red_title_surf, red_title_rect)
    screen.blit(rest_title_surf, rest_title_rect)

    # Khu vực ảnh nén lossless rle
    pygame.draw.rect(screen, BLACK, (20, 90, 480, 730), 2)
    pygame.draw.rect(screen, BLACK, (160, 140, 200, 220), 2)
    pygame.draw.line(screen, BLACK, (20, 410), (1454, 410), 2)
    lossless_title = font.render("Ảnh nén lossless RLE", True, BLACK)
    screen.blit(lossless_title, (180, 420))
    if lossless_compressed is not None and compressed_image_RLE is not None:
        # Placeholder cho ảnh nén (có thể thêm logic hiển thị ảnh giải nén)
        screen.blit(compressed_image_RLE, (160,140))
        pass
    lossless_info_y = 495
    lossless_info_texts = [
        f"Tên file nén: RLE_encoding_file_{image_name.split('.')[0]}.npz" if lossless_compressed is not None else "Tên file nén:",
        f"Kích cỡ nén: {np.array(lossless_compressed, dtype=np.uint8).nbytes} bytes (~{np.array(lossless_compressed, dtype=np.uint8).nbytes / 1024:.2f} KB)"
        if lossless_compressed is not None else "Kích cỡ nén:",
        f"Giảm tỉ lệ: {((original_size - np.array(lossless_compressed, dtype=np.uint8).nbytes) / original_size * 100):.2f}% (Tỉ lệ nén: {original_size / np.array(lossless_compressed, dtype=np.uint8).nbytes:.1f}:1)"
        if lossless_compressed is not None and original_size > 0 else "Giảm tỉ lệ:"
    ]
    for text in lossless_info_texts:
        text = shorten_text(text)
        info_surface = info_font.render(text, True, BLACK)
        screen.blit(info_surface, (70, lossless_info_y))
        lossless_info_y += 80

    # Khu vực ảnh nén lossy rle dropbit
    pygame.draw.rect(screen, BLACK, (498, 90, 480, 730), 2)
    pygame.draw.rect(screen, BLACK, (638, 140, 200, 220), 2)
    lossy_title = font.render("Ảnh nén lossy RLE dropbit", True, BLACK)
    screen.blit(lossy_title, (642, 420))
    if lossy_compressed_dropbit is not None and compressed_image_LRLE_dropbit is not None:
        # Placeholder cho ảnh nén (có thể thêm logic hiển thị ảnh giải nén)
        screen.blit(compressed_image_LRLE_dropbit, (638, 140))
        pass
    lossy_info_y = 495
    lossy_info_texts = [
        f"Tên file nén: LRLE_dropbit_encoding_file_{image_name.split('.')[0]}.npz" if lossy_compressed_dropbit is not None else "Tên file nén:",
        f"Kích cỡ nén: {np.array(lossy_compressed_dropbit, dtype=np.uint8).nbytes} bytes "
        f"(~{np.array(lossy_compressed_dropbit, dtype=np.uint8).nbytes / 1024:.2f} KB)"
        if lossy_compressed_dropbit is not None else "Kích cỡ nén:",
        f"Giảm tỉ lệ: {((original_size - np.array(lossy_compressed_dropbit, dtype=np.uint8).nbytes) / original_size * 100):.2f}% (Tỉ lệ nén: {original_size / np.array(lossy_compressed_dropbit, dtype=np.uint8).nbytes:.1f}:1)"
        if lossy_compressed_dropbit is not None and original_size > 0 else "Giảm tỉ lệ:"
    ]

    for text in lossy_info_texts:
        text = shorten_text(text)
        info_surface = info_font.render(text, True, BLACK)
        screen.blit(info_surface, (548, lossy_info_y))
        lossy_info_y += 80

    # Khu vực ảnh nén lossy blending lossy rle
    pygame.draw.rect(screen, BLACK, (976, 90, 480, 730), 2)
    pygame.draw.rect(screen, BLACK, (1116, 140, 200, 220), 2)
    lossy_title = font.render("Ảnh nén lossy RLE blending", True, BLACK)
    screen.blit(lossy_title, (1095, 420))
    if lossy_compressed_dropbit is not None and compressed_image_LRLE_blending is not None:
        # Placeholder cho ảnh nén (có thể thêm logic hiển thị ảnh giải nén)
        screen.blit(compressed_image_LRLE_blending, (1116, 140))
        pass
    lossy_info_y = 495
    lossy_info_texts = [
        f"Tên file nén: LRLE_blending_encoding_file_{image_name.split('.')[0]}.npz" if lossy_compressed_blending is not None else "Tên file nén:",
        f"Kích cỡ nén: {np.array(lossy_compressed_blending, dtype=np.uint8).nbytes} bytes "
        f"(~{np.array(lossy_compressed_blending, dtype=np.uint8).nbytes / 1024:.2f} KB)"
        if lossy_compressed_blending is not None else "Kích cỡ nén:",
        f"Giảm tỉ lệ: {((original_size - np.array(lossy_compressed_blending, dtype=np.uint8).nbytes) / original_size * 100):.2f}% (Tỉ lệ nén: {original_size / np.array(lossy_compressed_blending, dtype=np.uint8).nbytes:.1f}:1)"
        if lossy_compressed_blending is not None and original_size > 0 else "Giảm tỉ lệ:"
    ]
    for text in lossy_info_texts:
        text = shorten_text(text)
        info_surface = info_font.render(text, True, BLACK)
        screen.blit(info_surface, (1026, lossy_info_y))
        lossy_info_y += 80

    # Khu vực ảnh gốc và điều khiển
    pygame.draw.rect(screen, BLACK, (1454, 90, 480, 730), 2)
    pygame.draw.rect(screen, BLACK, (1594, 220, 200, 220), 2)
    if uploaded_image:
        screen.blit(uploaded_image, (1594, 220))
    image_name_text = font.render(f"Tên hình ảnh: {shorten_text(image_name)}", True, BLACK) if image_name else font.render("Tên hình ảnh:", True, BLACK)
    color_format_text = "(RGB)" if is_rgb else "(Grayscale)"
    image_size_text = font.render(f"Kích thước: {original_size} bytes (~{original_size/1024:.2f} KB) {color_format_text}" if original_size > 0 else "Kích thước ban đầu:", True, BLACK)
    screen.blit(image_name_text, (1494, 130))
    screen.blit(image_size_text, (1494, 170))

    # Nút điều hướng
    prev_button_rect = pygame.Rect(1544, 310, 40, 40)
    next_button_rect = pygame.Rect(1804, 310, 40, 40)
    pygame.draw.rect(screen, HOVER_COLOR if prev_button_rect.collidepoint(pygame.mouse.get_pos()) else GRAY, prev_button_rect)
    pygame.draw.rect(screen, HOVER_COLOR if next_button_rect.collidepoint(pygame.mouse.get_pos()) else GRAY, next_button_rect)
    prev_text = button_font.render("<", True, BLACK)
    next_text = button_font.render(">", True, BLACK)
    screen.blit(prev_text, prev_text.get_rect(center=prev_button_rect.center))
    screen.blit(next_text, next_text.get_rect(center=next_button_rect.center))

# Nút chức năng
    upload_button_rect = pygame.Rect(1594, 480, 200, 40)
    run_button_rect = pygame.Rect(1594, 560, 200, 40)
    folder_button_rect = pygame.Rect(1594, 640, 200, 40)
    upload_encoded_button_rect = pygame.Rect(1594, 720, 200, 40)
    pygame.draw.rect(screen, HOVER_COLOR if upload_button_rect.collidepoint(pygame.mouse.get_pos()) else GRAY, upload_button_rect)
    pygame.draw.rect(screen, HOVER_COLOR if run_button_rect.collidepoint(pygame.mouse.get_pos()) else GRAY, run_button_rect)
    pygame.draw.rect(screen, HOVER_COLOR if folder_button_rect.collidepoint(pygame.mouse.get_pos()) else GRAY, folder_button_rect)
    pygame.draw.rect(screen, HOVER_COLOR if upload_encoded_button_rect.collidepoint(pygame.mouse.get_pos()) else GRAY, upload_encoded_button_rect)
    upload_text = button_font.render("Tải hình ảnh", True, BLACK)
    run_text = button_font.render("Chạy thuật toán", True, BLACK)
    folder_text = button_font.render("Tải thư mục", True, BLACK)
    uploadencode_text = button_font.render("Tải file mã hóa", True, BLACK)
    screen.blit(upload_text, upload_text.get_rect(center=upload_button_rect.center))
    screen.blit(run_text, run_text.get_rect(center=run_button_rect.center))
    screen.blit(folder_text, folder_text.get_rect(center=folder_button_rect.center))
    screen.blit(uploadencode_text, uploadencode_text.get_rect(center=upload_encoded_button_rect.center))

    # Xử lý sự kiện
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if upload_button_rect.collidepoint(event.pos):
                image_files = []
                selected_folder = None
                upload_image()
            elif run_button_rect.collidepoint(event.pos):
                run_algorithm()
            elif folder_button_rect.collidepoint(event.pos):
                image_files = []
                uploaded_image = None
                image_name = ""
                original_size = 0
                upload_folder()
            elif prev_button_rect.collidepoint(event.pos):
                if image_files and current_image_index > 0:
                    current_image_index -= 1
                    load_image()
            elif next_button_rect.collidepoint(event.pos):
                if image_files and current_image_index < len(image_files) - 1:
                    current_image_index += 1
                    load_image()
            elif upload_encoded_button_rect.collidepoint(event.pos):
                image_files = []
                selected_folder = None
                upload_encoded_file()

    pygame.display.flip()
    pygame.time.delay(30)

pygame.quit()