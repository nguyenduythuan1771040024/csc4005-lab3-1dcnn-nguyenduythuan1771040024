# CSC4005 Lab 3 - UrbanSound8K Classification with 1D-CNN

Repository này là bài làm Lab 3 cho bài toán phân loại âm thanh môi trường trên bộ dữ liệu UrbanSound8K. Pipeline chính dùng MFCC/log-mel theo thời gian và mô hình 1D-CNN. Phần raw waveform 1D-CNN được chạy thêm như bài mở rộng để so sánh.

## 1. Nội dung đã hoàn thành

- Đọc metadata và audio theo cấu trúc UrbanSound8K.
- Chia dữ liệu theo fold:
  - train: fold 1-8
  - validation: fold 9
  - test: fold 10
- Chuẩn hóa audio về mono, `16000 Hz`, độ dài `4.0` giây.
- Trích xuất MFCC, log-mel hoặc raw waveform.
- Huấn luyện 1D-CNN với BatchNorm, ReLU, MaxPool và Dropout.
- Lưu `metrics.json`, `history.csv`, `curves.png`, `confusion_matrix.png`.
- Chạy W&B ở chế độ offline để có thể sync sau.
- Viết báo cáo kết quả trong `REPORT.md`.

## 2. Cấu trúc chính

```text
.
├── configs/
│   ├── baseline_mfcc_1dcnn.json
│   ├── extension_raw_waveform.json
│   └── fast_debug.json
├── docs/
│   ├── GITHUB_CLASSROOM_GUIDE.md
│   ├── LAB_GUIDE_LAB3.md
│   ├── RUBRIC.md
│   └── WANDB_GUIDE.md
├── outputs/
│   ├── 1771040024_mfcc_1dcnn_baseline/
│   ├── 1771040024_logmel_1dcnn/
│   └── 1771040024_raw_waveform_extension/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── ci/
│   ├── check_structure.py
│   └── smoke_train.py
├── REPORT.md
├── REPORT_TEMPLATE.md
├── requirements.txt
└── README.md
```

Dataset UrbanSound8K, file `.rar`, cache, W&B local log và checkpoint `.pt` không được commit lên GitHub.

## 3. Cài đặt môi trường

Khuyến nghị dùng Python 3.10 hoặc 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Trong lần chạy thực tế, môi trường Conda đã dùng là:

```text
C:\Users\nguye\.conda\envs\DL\python.exe
```

## 4. Chuẩn bị dữ liệu

Giải nén UrbanSound8K sao cho có cấu trúc:

```text
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   └── ...
└── metadata/
    └── UrbanSound8K.csv
```

Khi chạy lệnh train, truyền đường dẫn qua `--data_dir`.

## 5. Chạy kiểm tra nhanh

```bash
python ci/check_structure.py
python ci/smoke_train.py
```

Kết quả mong đợi:

```text
Structure OK
Smoke train OK
```

## 6. Chạy các thí nghiệm

### MFCC baseline

```bash
python -m src.train ^
  --config configs/baseline_mfcc_1dcnn.json ^
  --data_dir UrbanSound8K ^
  --run_name 1771040024_mfcc_1dcnn_baseline ^
  --wandb_mode offline
```

### Log-mel extension

```bash
python -m src.train ^
  --config configs/baseline_mfcc_1dcnn.json ^
  --data_dir UrbanSound8K ^
  --feature_type logmel ^
  --run_name 1771040024_logmel_1dcnn ^
  --wandb_mode offline
```

### Raw waveform extension

```bash
python -m src.train ^
  --config configs/extension_raw_waveform.json ^
  --data_dir UrbanSound8K ^
  --run_name 1771040024_raw_waveform_extension ^
  --wandb_mode offline
```

## 7. Kết quả đã chạy

| Pipeline | Best val acc | Test acc | Avg epoch time |
|---|---:|---:|---:|
| MFCC + 1D-CNN | 0.6091 | 0.5097 | 6.07 s |
| log-mel + 1D-CNN | 0.6631 | 0.6065 | 6.63 s |
| raw waveform + 1D-CNN | 0.5832 | 0.6022 | 33.11 s |

Trong các thí nghiệm đã chạy, log-mel + 1D-CNN là hướng tốt nhất vì có test accuracy cao nhất và thời gian train gần MFCC hơn nhiều so với raw waveform.

## 8. Output nộp bài

Các thư mục output đã lưu:

```text
outputs/1771040024_mfcc_1dcnn_baseline/
outputs/1771040024_logmel_1dcnn/
outputs/1771040024_raw_waveform_extension/
```

Mỗi thư mục nộp các file:

- `metrics.json`
- `history.csv`
- `curves.png`
- `confusion_matrix.png`
- `used_config.json`

Checkpoint `best_model.pt` được tạo khi train nhưng không commit vì là file nhị phân lớn và không bắt buộc trong rubric.

## 9. W&B

Các run được tạo ở chế độ offline. Sau khi đăng nhập W&B, sync bằng:

```bash
wandb sync wandb/offline-run-20260514_091110-bklv1ims
wandb sync wandb/offline-run-20260514_091254-shu77u10
wandb sync wandb/offline-run-20260514_091503-un1cg0lh
```

Sau khi sync, dán link W&B project hoặc run vào `REPORT.md`.

## 10. Báo cáo

Báo cáo chính nằm ở:

```text
REPORT.md
```

Báo cáo đã phân tích:

- cấu hình tiền xử lý,
- kiến trúc 1D-CNN,
- learning curves,
- confusion matrix,
- các lớp dễ nhầm,
- so sánh MFCC, log-mel và raw waveform.
