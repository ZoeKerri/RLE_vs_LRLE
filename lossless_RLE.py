import cv2
import numpy as np
import os
from RLE_LRLE_utilities import Utilities

class RLE:
    """
        Cách 1 đảm bảo dữ liệu lưu luôn là uint8 phù hợp với hình ảnh ít có lặp lại, số lần đếm tối đa chỉ có 0->255
        Mảng compressed ở trên được lưu dưới dạng mảng 1 chiều gồm 4 giá trị. 3 giá trị đầu là 3 kênh màu, giá trị 4 là số lần lặp.
        Đối với ảnh xám, mảng compressed gồm 2 giá trị: 1 giá trị là mức xám, giá trị thứ 2 là số lần lặp.

        Cách 2 giống với cách 1 khác mỗi cái là số lần đếm k giới hạn do đó kiểu dữ liệu có thể là uint16 32 hay 64 gì đó.
    """
    
    def __init__(self, best_method_name=0):
        self.best_method_name = best_method_name
        """
            best_method_name có 5 giá trị
            0: khởi đầu, 1: ảnh xám - dùng method 1 để nén, 2: ảnh màu - dùng method 1 để nén
            3 và 4 thì tương tự trên mà dùng method 2
        """
        

    def _determine_dtype(self, max_val):
        if max_val <= np.iinfo(np.uint8).max:
            return np.uint8
        elif max_val <= np.iinfo(np.uint16).max:
            return np.uint16
        elif max_val <= np.iinfo(np.uint32).max:
            return np.uint32
        else:
            return np.uint64

    def rle_compressing_method_1(self, data, is_grayscale=False):
        """
        Args:
            data: Hình ảnh
            is_grayscale: hình màu hay hình trắng đen

        Returns:
            compressed: dữ liệu đã được nén
        """
        if is_grayscale:
            # Với ảnh xám, chỉ có 1 kênh, làm phẳng thành mảng 1 chiều
            flattened = data.flatten()
            compressed = []
            current_value = flattened[0]
            count = 1
            for pixel in flattened[1:]:
                if pixel == current_value and count < 255:
                    count += 1
                else:
                    compressed.append(current_value)
                    compressed.append(count)
                    current_value = pixel
                    count = 1
            compressed.append(current_value)
            compressed.append(count)
            return compressed
        else:
            # lý do phải làm phẳng z mục đích là để lấy pixel dễ hơn vd ta có 20x20x3 thì ta thu được mảng 2 chiều là (20*20) x 3
            flattened = data.reshape(-1, 3)
            compressed = []
            current_value = flattened[0]
            count = 1
            for pixel in flattened[1:]:
                # lý do count<255 là vì để tiết kiệm dữ liệu thì ta sẽ lưu trữ bằng kiểu dữ liệu uint8 chiếm 1byte
                # 1 byte = 8 bit => 8 bit thì chứa được giá trị cao nhất là 255
                # màu RGB luôn nằm trong 0->255 nên ta không lo, cái cần lo là count nên ta đặt điều kiện <255 ở đây để đảm bảo kiểu dữ liệu khi tính qua thư viện numpy là uint8
                if np.array_equal(pixel, current_value) and count < 255:
                    count += 1
                else:
                    # extend phương thức thêm đối tượng vào cuối mảng có điều là nó làm cái danh sách nó phẳng
                    # vd: arr = [1,2,3] temp = [4,5] arr.append(temp) => arr có giá trị [1,2,3,[4,5]]
                    # vd: extend thì làm phẳng temp cho temp vào như là append từng giá trị con 1 thay vì append như 1 list
                    # vd arr = [1,2,3] temp = [4,5], arr.extend(temp) => arr = [1,2,3,4,5]
                    compressed.extend(current_value.tolist())
                    compressed.append(count)
                    current_value = pixel
                    count = 1
            compressed.extend(current_value.tolist())
            compressed.append(count)
            return compressed

    # phù hợp với ảnh có sự lặp lại nhiều
    # Cách 2 giống với cách 1 khác mỗi cái là số lần đếm k giới hạn
    def rle_compressing_method_2(self, data, is_grayscale=False):
        """
        Args:
            data: Hình ảnh
            is_grayscale: hình màu hay hình trắng đen

        Returns:
            compressed: dữ liệu đã được nén, max_val: giá trị count lớn nhất trong hàm dùng để xác định kiểu dữ liệu để nén như uint8 16...
        """
        if is_grayscale:
            flattened = data.flatten()
            compressed = []
            current_value = flattened[0]
            count = 1
            max_val = 1
            for pixel in flattened[1:]:
                if pixel == current_value:
                    count += 1
                else:
                    compressed.append(current_value)
                    compressed.append(count)
                    if max_val < count:
                        max_val = count
                    current_value = pixel
                    count = 1
            if max_val < count:
                max_val = count
            compressed.append(current_value)
            compressed.append(count)
            return compressed, max_val
        else:
            flattened = data.reshape(-1, 3)
            compressed = []
            current_value = flattened[0]
            count = 1
            max_val = 1
            for pixel in flattened[1:]:
                if np.array_equal(pixel, current_value):
                    count += 1
                else:
                    compressed.extend(current_value.tolist())
                    compressed.append(count)
                    if max_val < count:
                        max_val = count
                    current_value = pixel
                    count = 1
            if max_val < count:
                max_val = count
            compressed.extend(current_value.tolist())
            compressed.append(count)
            return compressed, max_val

    #Hàm dùng để test hiệu suất 
    def test_RLE1_vs_RLE2(self):
        # Đọc ảnh
        image_path = 'test_RLE/anh_chua_nen_lap_lai_nhieu.bmp'
        if not os.path.exists(image_path):
            print(f"Không tìm thấy ảnh: {image_path}. Vui lòng đảm bảo file ảnh tồn tại.")
            return None

        bgr_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if bgr_img is None:
            return

        is_grayscale = len(bgr_img.shape) == 2 or (len(bgr_img.shape) == 3 and bgr_img.shape[2] == 1)
        if is_grayscale and len(bgr_img.shape) == 3:
            img = bgr_img[:, :, 0]  # Lấy kênh đầu tiên nếu ảnh có 3 chiều
        elif is_grayscale:
            img = bgr_img  # Ảnh xám không cần chuyển màu
        else:
            img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # Tính kích thước mảng (tính theo byte)
        original_size_bytes = img.nbytes
        original_size_kb = original_size_bytes / 1024
        print(f"Kích thước ảnh gốc: {original_size_bytes} bytes ~ {original_size_kb:.2f} KB")

        # Nén ảnh bằng RLE
        if is_grayscale:
            compressed_method_1_arr = self.rle_compressing_method_1(img, is_grayscale=True)
            compressed_np_method_1 = np.array(compressed_method_1_arr, dtype=np.uint8)
            compressed_size_bytes_method_1 = compressed_np_method_1.nbytes
            compressed_size_kb_method_1 = compressed_size_bytes_method_1 / 1024

            compressed_method_2_arr, max_val = self.rle_compressing_method_2(img, is_grayscale=True)
            dtype_method_2 = self._determine_dtype(max_val)
            compressed_np_method_2 = np.array(compressed_method_2_arr, dtype=dtype_method_2)
            compressed_size_bytes_method_2 = compressed_np_method_2.nbytes
            compressed_size_kb_method_2 = compressed_size_bytes_method_2 / 1024

            print(f"Kích thước sau nén bằng cách 1 (xám): {compressed_size_bytes_method_1} bytes ~ {compressed_size_kb_method_1:.2f} KB")
            print(f"Kích thước sau nén bằng cách 2 (xám): {compressed_size_bytes_method_2} bytes ~ {compressed_size_kb_method_2:.2f} KB")

            if compressed_size_bytes_method_1 < compressed_size_bytes_method_2:
                print("\nPhương pháp nén 1 (xám) cho kích thước nhỏ hơn.")
                best_compressed_data = compressed_method_1_arr
                self.best_method_name = 2
                decompressed_img = Utilities.rle_lrle_decompressing(best_compressed_data, img.shape, self.best_method_name)
                output_filename = "test_RLE/anh_nen_tot_nhat_method_1_gray.bmp"
            else:
                print("\nPhương pháp nén 2 (xám) cho kích thước nhỏ hơn hoặc bằng.")
                best_compressed_data = compressed_method_2_arr
                self.best_method_name = 4
                decompressed_img = Utilities.rle_lrle_decompressing(best_compressed_data, img.shape, self.best_method_name)
                output_filename = "test_RLE/anh_nen_tot_nhat_method_2_gray.bmp"

            cv2.imwrite(output_filename, decompressed_img)
        else:
            compressed_method_1_arr = self.rle_compressing_method_1(img)
            compressed_np_method_1 = np.array(compressed_method_1_arr, dtype=np.uint8)
            compressed_size_bytes_method_1 = compressed_np_method_1.nbytes
            compressed_size_kb_method_1 = compressed_size_bytes_method_1 / 1024

            compressed_method_2_arr, max_val = self.rle_compressing_method_2(img)
            dtype_method_2 = self._determine_dtype(max_val)
            compressed_np_method_2 = np.array(compressed_method_2_arr, dtype=dtype_method_2)
            compressed_size_bytes_method_2 = compressed_np_method_2.nbytes
            compressed_size_kb_method_2 = compressed_size_bytes_method_2 / 1024

            print(f"Kích thước sau nén bằng cách 1 (màu): {compressed_size_bytes_method_1} bytes ~ {compressed_size_kb_method_1:.2f} KB")
            print(f"Kích thước sau nén bằng cách 2 (màu): {compressed_size_bytes_method_2} bytes ~ {compressed_size_kb_method_2:.2f} KB")

            if compressed_size_bytes_method_1 < compressed_size_bytes_method_2:
                print("\nPhương pháp nén 1 (màu) cho kích thước nhỏ hơn.")
                best_compressed_data = compressed_method_1_arr
                self.best_method_name = 1
                decompressed_img = Utilities.rle_lrle_decompressing(best_compressed_data, img.shape, self.best_method_name)
                output_filename = "test_RLE/anh_nen_tot_nhat_method_1_color.bmp"
            else:
                print("\nPhương pháp nén 2 (màu) cho kích thước nhỏ hơn hoặc bằng.")
                best_compressed_data = compressed_method_2_arr
                self.best_method_name = 3
                decompressed_img = Utilities.rle_lrle_decompressing(best_compressed_data, img.shape, self.best_method_name)
                output_filename = "test_RLE/anh_nen_tot_nhat_method_2_color.bmp"

            decompressed_bgr = cv2.cvtColor(decompressed_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_filename, decompressed_bgr)

        print(f"Ảnh đã giải nén từ phương pháp tốt nhất đã được lưu thành: {output_filename}")
        return

    #Hàm dùng để so sánh xem cách nào nén bên lossless ít dữ liệu hơn sau đó thì chuyển qua bên lossy để nén tiếp (nếu có)
    def get_compressed_data(self, image_path):
        if not os.path.exists(image_path):
            print(f"Không tìm thấy ảnh: {image_path}. Vui lòng đảm bảo file ảnh tồn tại.")
            return None, None

        bgr_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if bgr_img is None:
            return None, None

        img_shape = bgr_img.shape
        print(f"Shape đọc từ OpenCV: {img_shape}")
        is_grayscale = len(bgr_img.shape) == 2 or (len(bgr_img.shape) == 3 and bgr_img.shape[2] == 1)
        if is_grayscale and len(bgr_img.shape) == 3:
            img = bgr_img[:, :, 0]
        elif is_grayscale:
            img = bgr_img
        else:
            img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        compressed_method_1_arr = self.rle_compressing_method_1(img, is_grayscale=is_grayscale)
        compressed_np_method_1 = np.array(compressed_method_1_arr, dtype=np.uint8)
        compressed_size_bytes_method_1 = compressed_np_method_1.nbytes

        compressed_method_2_arr, max_val = self.rle_compressing_method_2(img, is_grayscale=is_grayscale)
        dtype_method_2 = self._determine_dtype(max_val)
        compressed_np_method_2 = np.array(compressed_method_2_arr, dtype=dtype_method_2)
        compressed_size_bytes_method_2 = compressed_np_method_2.nbytes

        if compressed_size_bytes_method_1 < compressed_size_bytes_method_2:
            self.best_method_name = "1" if not is_grayscale else "2"
            return compressed_method_1_arr, img.shape, 255 if not is_grayscale else np.iinfo(np.uint8).max
        else:
            self.best_method_name = "3" if not is_grayscale else "4"
            return compressed_method_2_arr, img.shape, max_val

if __name__ == "__main__":
    rle_instance = RLE("")
    rle_instance.test_RLE1_vs_RLE2()
    print(f"\nPhương pháp nén tốt nhất đã sử dụng là: {rle_instance.best_method_name}")