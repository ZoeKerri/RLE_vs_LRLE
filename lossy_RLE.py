from lossless_RLE import RLE
import numpy as np


class Lossy_RLE:
    """
        method_name 1 2 3 4 là gì?
        method_name là kế thừa từ sau khi nén lossless RLE
        Và 1 3 tượng trưng cho ảnh màu nén cách 1 và 2 bên RLE
        2 4 tượng trưng cho ảnh xám nén cách 1 và 2 bên RLE
        Chia vậy là do kiểu dữ liệu trả về của 2 ảnh nó khác nhau và kiểu nén RLE cũng khác nhau nên phải có method_name để nhận kiểu dữ liệu cho đúng.
        Còn cách 1, cách 2 nén RLE là gì thì bên file lossless_RLE có giải thích.
    """
    def _get_dtype(self, max_val):
        #xác định kiểu dữ liệu để lưu
        if max_val <= np.iinfo(np.uint8).max:
            return np.uint8
        elif max_val <= np.iinfo(np.uint16).max:
            return np.uint16
        elif max_val <= np.iinfo(np.uint32).max:
            return np.uint32
        else:
            return np.uint64

    def rle_lossy_compressing_dropbit(self, compressed_data, method_name, max_val, drop_bits=4, short_run_threshold=2):
        """
        Nén RLE lossy bằng cách drop bit để tăng khả năng lặp lại.
        Nếu short run sau khi drop bit giống run trước (dropbit tạm không lưu), gộp bằng cách tăng count của run trước.
        Args:
            compressed_data: Dữ liệu đã nén RLE (numpy array).
            method_name: Phương thức nén ("1": màu cách 1, "2": xám cách 1, "3": màu cách 2, "4": xám cách 2).
            max_val: Giá trị đếm lớn nhất từ RLE gốc.
            drop_bits: Số bit bị drop (mặc định 4).
            short_run_threshold: Ngưỡng độ dài lặp ngắn (mặc định 2).
        Returns:
            compressed_data: Dữ liệu nén lossy (numpy array).
            max_val: Giá trị đếm lớn nhất trong dữ liệu nén.
        """
        # Xác định ảnh xám hay màu
        is_grayscale = method_name in ["2", "4"]
        step = 2 if is_grayscale else 4  # Ảnh xám: [giá_trị, đếm], màu: [R,G,B,đếm]

        # Validate input
        if len(compressed_data) % step != 0:
            raise ValueError(f"Dữ liệu sai: {len(compressed_data)}. Dữ liệu phải chia hết cho step {step}.")
        if len(compressed_data) == 0:
            return np.array([], dtype=np.uint8), max_val

        # Tạo mặt nạ để drop bit
        mask = 0xFF << drop_bits  # Dịch trái để giữ bit cao, ví dụ drop_bits=4 -> mask=11110000

        lossy_compressed = []
        i = 0

        while i < len(compressed_data):
            count = compressed_data[i + step - 1]  # Lấy giá trị đếm
            if count > max_val:
                max_val = count

            if count >= short_run_threshold:
                # Long run: Giữ nguyên
                lossy_compressed.extend(compressed_data[i:i+step])
                i += step
                continue

            # Short run: Áp dụng drop bit
            if is_grayscale:
                gray = compressed_data[i]
                lossy_value = gray & mask  # Drop bit 
                current_run = [lossy_value, count]
            else:
                rgb = compressed_data[i:i+3]
                lossy_value = [c & mask for c in rgb]  # Drop bit cho từng kênh R,G,B
                current_run = lossy_value + [count]

            # Kiểm tra nếu run trước tồn tại và có giá trị giống sau drop bit
            if lossy_compressed and len(lossy_compressed) >= step:
                prev_value = lossy_compressed[-step:-1]  # Giá trị của run trước (không lấy count)
                #dropbit tạm cho long run trước đó xem 2 giá trị có bằng nhau không
                if is_grayscale:
                    prev_value = prev_value[0] & mask
                else:
                    temp_value = [c & mask for c in prev_value]
                    prev_value = temp_value
                if prev_value == (lossy_value if not is_grayscale else [lossy_value]):
                    # Gộp: Tăng count của run trước
                    lossy_compressed[-1] += count
                    if lossy_compressed[-1] > max_val:
                        max_val = lossy_compressed[-1]
                else:
                    # Không gộp được: Thêm run mới
                    lossy_compressed.extend(current_run)
            else:
                # Không có run trước: Thêm run mới
                lossy_compressed.extend(current_run)

            i += step

        # Chuyển thành numpy array với dtype phù hợp
        dtype = self._get_dtype(max_val)
        return np.array(lossy_compressed, dtype=dtype), max_val

    #Blending
    def rle_lossy_blending_short_runs(self, compressed_data, method_name, max_val, short_run_threshold=3, blending_threshold = 5):
        """
        Nén RLE lossy bằng cách gộp các short run: Gọi thuật toán này là Blending
        Gộp các short run liên tiếp thành một run với giá trị đầu tiên, cuối hoặc giữa.
        Args:
            compressed_data: Dữ liệu đã nén RLE (numpy array).
            method_name: Phương thức nén ("1": màu cách 1, "2": xám cách 1, "3": màu cách 2, "4": xám cách 2).
            max_val: Giá trị đếm lớn nhất từ RLE gốc.
            short_run_threshold: Ngưỡng độ dài lặp ngắn (mặc định 3).
        Returns:
            compressed_data: Dữ liệu nén lossy (numpy array).
            max_val: Giá trị đếm lớn nhất trong dữ liệu nén.
        """
        is_grayscale = method_name in ["2", "4"]
        step = 2 if is_grayscale else 4

        # Validate input
        if len(compressed_data) % step != 0:
            raise ValueError(f"Invalid compressed_data length: {len(compressed_data)}. Must be multiple of {step}.")
        if len(compressed_data) == 0:
            return np.array([], dtype=np.uint8), max_val

        lossy_compressed = []
        i = 0

        while i < len(compressed_data):
            count = compressed_data[i + step - 1]
            if count > max_val:
                max_val = count

            if count >= short_run_threshold:
                # Long run: Giữ nguyên
                lossy_compressed.extend(compressed_data[i:i+step])
                i += step
                continue

            #Trường hợp phần tử đầu tiên là short run
            if len(lossy_compressed) == 0:
                next_count = compressed_data[i + 2*step - 1]
                if count >= short_run_threshold:
                    lossy_compressed.extend(compressed_data[i+step:i+2*step-1],count+next_count)
                    i += step*2
                    continue

            # Vì đã loại đc phần tử đầu tiên là short run nên chắc chắn phần tử tiếp theo phần tử short run là long run hoặc phần tử đầu tiên chính là long run

            # các short run sẽ được chứ vào mảng tạm và tiến hành tính toán
            temp_runs = []
            current_i = i
            while current_i < len(compressed_data):
                current_count = compressed_data[current_i + step - 1]
                if current_count >= short_run_threshold or len(temp_runs) >= blending_threshold:
                    break
                value = compressed_data[current_i] if is_grayscale else compressed_data[current_i:current_i+3]
                temp_runs.append([value, current_count])
                current_i += step

            if temp_runs:
                total_count = sum(run[1] for run in temp_runs)
                if total_count > max_val:
                    max_val = total_count

                # Xác định phần tử nằm giữa sẽ làm đại diện cho short run
                mid_idx = len(temp_runs) // 2
                if len(temp_runs) % 2 == 0 and len(temp_runs) > 1:
                    mid_value1 = temp_runs[mid_idx-1][0]
                    mid_value2 = temp_runs[mid_idx][0]
                    if is_grayscale:
                        mid_value = int((mid_value1 + mid_value2) / 2)
                    else:
                        mid_value = ((np.array(mid_value1) + np.array(mid_value2)) / 2).astype(int).tolist()
                else:
                    mid_value = temp_runs[mid_idx][0]

                if is_grayscale:
                    lossy_compressed.extend([mid_value, total_count])
                else:
                    lossy_compressed.extend(mid_value)
                    lossy_compressed.append(total_count)

            i = current_i

        dtype = self._get_dtype(max_val)
        return np.array(lossy_compressed, dtype=dtype), max_val

    def get_LRLE_method(self, compressed_data, method_name, max_val_rle, drop_bits=4, short_run_threshold=3, blending_threshold = 5):
        """
        So sánh hai phương pháp nén lossy (drop bit và merge short runs) và trả về dữ liệu nén của phương pháp tốt hơn.
        Args:
            compressed_data: Dữ liệu đã nén bằng RLE
            method_name: Phương thức nén ("1": màu cách 1, "2": xám cách 1, "3": màu cách 2, "4": xám cách 2).
            drop_bits: Số bit bị drop cho phương pháp drop bit (mặc định 4).
            short_run_threshold: Ngưỡng độ dài lặp ngắn (mặc định 3).
        Returns:
            compressed_merge/compressed_drop: Dữ liệu nén của phương pháp tốt nhất (numpy array).
            max_val: Giá trị đếm lớn nhất.
            best_method: Tên phương pháp tốt nhất ("Drop Bit" hoặc "Merge Short Runs").
        """
        lrle_instance = Lossy_RLE()

        compressed_drop, max_val_drop =  lrle_instance.rle_lossy_compressing_dropbit(
            compressed_data, method_name, max_val_rle,  drop_bits, short_run_threshold
        )
        compressed_merge, max_val_merge =  lrle_instance.rle_lossy_blending_short_runs(
            compressed_data, method_name, max_val_rle, short_run_threshold, blending_threshold
        )
        
        size_drop = compressed_drop.nbytes
        size_merge = compressed_merge.nbytes
        
        if size_drop <= size_merge:
            return compressed_drop, max_val_drop, "Drop Bit"
        else:
            return compressed_merge, max_val_merge, "Merge Short Runs"
        