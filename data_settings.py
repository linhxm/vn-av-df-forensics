"""CHỌN PART ĐANG LÀM Ở LOCAL - chỉnh PART trước khi chạy các bước data.

Ví dụ PART = 1: 01_collect ... 05_export cùng đọc/ghi vn-av-df-data-part1.
PART = 2: các bước chuyển sang part2; 01_collect tạo videos.csv rỗng nếu chưa có.
CSV chỉ chứa URL/ID của part đang chọn, không tự quyết định số part.
Giữ PART cố định trong khi chạy hết chuỗi bước của part đó.

Review/finalize generation ở local cũng dùng PART này.
Kaggle: chọn PART ngay trong generate.ipynb; chọn DATASET_PARTS trong train.ipynb.
Không cần sửa file này trên Kaggle vì notebook ghi đè cấu hình trong phiên chạy.
"""

DATASET_NAME = "vn-av-df-data"
PART = 1  # Ví dụ: 1 -> part1; 2 -> part2. Không đổi ID người/nguồn theo số part.


def part_name(number=None):
    """Part là gói dữ liệu bổ sung, không phải số phiên bản của thí nghiệm."""
    number = PART if number is None else number
    if type(number) is not int or number < 1:
        raise ValueError("PART must be a positive integer")
    return f"{DATASET_NAME}-part{number}"
