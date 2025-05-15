import numpy as np
import os

class Utilities:
    @staticmethod
    def rle_lrle_decompressing(compressed_data, original_shape, method_name):
        """
        Giải nén dữ liệu RLE thành ảnh gốc.
        Args:
            compressed_data: Mảng dữ liệu đã nén (list hoặc numpy array).
            original_shape: Shape của ảnh gốc (tuple).
            method_name: Phương thức nén ("1", "2", "3", hoặc "4").
        Returns:
            decompressed_array: Ảnh đã giải nén (numpy array).
        """
        decompressed = []
        if method_name in ["2", "4"]:  # Ảnh xám cách 1 hoặc cách 2
            length = len(compressed_data)
            for i in range(0, length, 2):
                gray = compressed_data[i]
                count = compressed_data[i+1]
                decompressed.extend([gray] * count)
            decompressed_array = np.array(decompressed, dtype=np.uint8)
            return decompressed_array.reshape(original_shape)
        else:  # Ảnh màu cách 1 hoặc cách 2

            length = len(compressed_data)
            decompressed = []
            for i in range(0, length, 4):
                rgb = compressed_data[i:i+3]
                count = compressed_data[i+3]
                decompressed.extend(list(rgb) * count)

            # Check the size of decompressed data
            total_pixels = len(decompressed) // 3
            expected_pixels = original_shape[0] * original_shape[1]

            if total_pixels != expected_pixels:
                raise ValueError(f"Decompressed data size ({total_pixels} pixels) does not match original_shape ({expected_pixels} pixels)")

            decompressed_array = np.array(decompressed, dtype=np.uint8).reshape(-1, 3)
            return decompressed_array.reshape(original_shape)
        
    @staticmethod
    def export_compressed_data(compressed_data, original_shape, method_name, output_path):
        """
        Xuất dữ liệu đã nén, shape gốc, và phương thức nén ra file .npz.
        Args:
            compressed_data: Mảng dữ liệu đã nén (list hoặc numpy array).
            original_shape: Shape của ảnh gốc (tuple).
            method_name: Phương thức nén ("1", "2", "3", hoặc "4").
            output_path: Đường dẫn file đầu ra (ví dụ: 'compressed_data.npz').
        """
        compressed_np = np.array(compressed_data)
        np.savez(output_path, 
                 compressed_data=compressed_np, 
                 original_shape=original_shape, 
                 method_name=method_name)
        print(f"Dữ liệu đã được xuất ra: {output_path}")

    @staticmethod
    def load_and_decompress(npz_file_path):
        """
        Đọc dữ liệu từ file .npz và giải nén.
        Args:
            npz_file_path: Đường dẫn đến file .npz chứa dữ liệu nén.
        Returns:
            decompressed_img: Ảnh đã giải nén (numpy array) hoặc None nếu lỗi.
        """
        if not os.path.exists(npz_file_path):
            print(f"Không tìm thấy file: {npz_file_path}. Vui lòng đảm bảo file tồn tại.")
            return None
        
        with np.load(npz_file_path, allow_pickle=True) as data:
            compressed_data = data['compressed_data']
            original_shape = data['original_shape']
            method_name = data['method_name']
        
        decompressed_img = Utilities.rle_lrle_decompressing(compressed_data, original_shape, method_name)
        print(f"Đã giải nén ảnh với phương thức: {method_name}")
        return decompressed_img, original_shape