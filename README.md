# Xử Lý Ảnh và Thị Giác Máy Tính — Bài tập lớn hợp nhất

Bài tập lớn môn Thị giác máy tính (121036): ba kỹ thuật độc lập xoay quanh một
câu hỏi chung — làm sao để một quyết định toàn cục (một ngưỡng, một bảng tra,
một ô lượng tử hóa) trở nên thích nghi với vị trí mà không phải trả giá bằng
thời gian tính toán.

Tài liệu đặc tả đầy đủ nằm trong file PDF ở thư mục `lab1/` (bản nháp cá nhân,
không commit lên repo này).

## 1. Ba kỹ thuật

| # | Kỹ thuật | File cài đặt | Người phụ trách |
|---|----------|--------------|-----------------|
| 1 | CLAHE: cân bằng histogram thích nghi có giới hạn tương phản | `clahe.ipynb` | Người 1 |
| 2 | Ngưỡng thích nghi bằng ảnh tích phân (Bradley–Roth, Sauvola) | `adaptive_threshold.py` | Người 2 |
| 3 | Biến đổi Hough đa tỉ lệ (thô-đến-mịn, bỏ phiếu theo gradient) | `hough_submission.py` | Người 3 |

Thí nghiệm minh họa (quét tham số, so sánh thuật toán, vẽ hình, đo thời gian)
do Người 4 và Người 5 thực hiện trong các notebook `notebooks/` trên branch
riêng, sau đó tích hợp thành một notebook nộp chung (xem mục 4).

## 2. Cấu trúc repo

```
.
├── README.md                    # file này
├── clahe.ipynb                  # Kỹ thuật 1 (Người 1)
├── adaptive_threshold.py        # Kỹ thuật 2 (Người 2)
├── test_adaptive_threshold.py   # Kiểm thử Kỹ thuật 2
├── hough_submission.py          # Kỹ thuật 3 (Người 3)
└── notebooks/                  # Notebook thí nghiệm (Người 4, 5, trên branch riêng)
```

## 3. Cài đặt môi trường

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib pillow jupyter
```

Chỉ dùng `numpy` (+ `time`, `matplotlib` ở script kiểm thử) cho phần cài đặt,
đúng danh sách cho phép trong PDF. Các hàm cấm (`cv2.integral`,
`cv2.adaptiveThreshold`, `cv2.createCLAHE`, `cv2.HoughLines`,
`skimage.filters.threshold_sauvola`, ...) chỉ được gọi trong ô đối chiếu có
ghi chú rõ, không dùng trong hàm nộp.

## 4. Cách dùng và kiểm thử

Kỹ thuật 2 (ví dụ chạy trên branch `feat/technique2-adaptive-threshold`):

```bash
python test_adaptive_threshold.py
```

Script kiểm thử so khớp với vòng lặp naive (sai số < 1e-9), kiểm tra tràn số
`int64`, kẹp phương sai không âm, và đo thời gian `local_mean_std` với
`radius` trong {1, 2, 4, 8, 16, 32, 64} để chứng minh độ phức tạp O(1).

```python
from adaptive_threshold import adaptive_threshold

out_bradley = adaptive_threshold(img, radius=15, method='bradley', t=0.15)
out_sauvola = adaptive_threshold(img, radius=15, method='sauvola', k=0.34)
```

## 5. Quy trình git của nhóm

- Mỗi kỹ thuật làm trên branch riêng: `feat/technique2-adaptive-threshold`,
  `feat/hough-experiments-report`, ...
- Đẩy branch lên rồi mở Pull Request vào `main` để review, không push thẳng.
- Trước khi xem code branch khác: `git status` phải sạch, rồi mới `git checkout`.
- Đồng bộ đầy đủ: `git fetch --all --prune`, tạo bản local còn thiếu bằng
  `git checkout -b <ten-local> origin/<ten-remote>`.

## 6. Quy định về AI (mục 5.1 trong PDF)

- **Phần cài đặt** (các file `.py`): tự viết, không dùng AI dưới mọi hình thức.
- **Phần thí nghiệm** (notebook): được dùng AI, nhưng mọi prompt phải chép
  nguyên văn vào notebook ngay trên ô mã tương ứng.

## 7. Nộp bài

Thầy yêu cầu nộp **một file `.ipynb` duy nhất kèm note**. Khi nộp, Người 5
tổng hợp code cả ba kỹ thuật và các thí nghiệm vào một notebook chạy được từ
đầu đến cuối (Kernel → Restart & Run All), rồi mới nộp.
