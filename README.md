# Movie Recommendation System - MiniProject

## Tổng quan
Đây là một pipeline hệ thống gợi ý phim tuần tự (sequence recommender) dùng mô hình RNN/LSTM/GRU để dự đoán phim kế tiếp trong lịch sử xem của người dùng. Mục tiêu: so sánh kiến trúc, tinh chỉnh optimizer/LR, và cung cấp demo tương tác.

## Yêu cầu
- Python 3.8+
- PyTorch
- pandas
- matplotlib
- ipywidgets (cho demo notebook)

Cài nhanh các package (ví dụ):

```bash
pip install torch pandas matplotlib ipywidgets
```

## Cấu trúc project (mô tả các file chính)
- `dataset.py`: tải và tiền xử lý dữ liệu (split theo thời gian, ánh xạ id -> index), chứa `MovieSequenceDataset` và `collate_fn`.
- `model.py`: định nghĩa `SequenceRecommender` (embedding + RNN/LSTM/GRU + head dự báo).
- `train.py`: hàm huấn luyện chính, bao gồm `train_with_oom_fallback` (tự động giảm batch size khi OOM xảy ra) và trả về `train_losses`, `val_losses`.
- `evaluate.py`: hàm `evaluate_model` để tính chỉ số đánh giá (Top-1, HR@10, ...).
- `train.py` (file): chứa logic huấn luyện chính (như mô tả trên).
- `modify_notebook.py`: trợ giúp hoặc script dùng để cập nhật/điều chỉnh notebook (nếu có).
- `Report_and_Demo.ipynb`: notebook chính để chạy toàn bộ pipeline: load dữ liệu, so sánh kiến trúc, tinh chỉnh optimizer/LR và demo tương tác. Notebook có cơ chế lưu cache huấn luyện (`training_cache.pt`) và khi cache tồn tại sẽ bỏ qua huấn luyện nhưng sẽ "replay" (in) các epoch đã lưu để người dùng vẫn thấy log.
- `Report_and_Demo copy.ipynb`: bản sao của notebook (backup).
- `training_cache.pt`: file cache (pickle) lưu kết quả huấn luyện (`results_arch`, `results_opt`) để tránh huấn luyện lại khi chạy toàn bộ notebook.
- `data/`: thư mục chứa dữ liệu (ví dụ `ml-1m/`).

## Hướng dẫn nhanh
1. Để huấn luyện thủ công (không dùng notebook):

```bash
python train.py
```

2. Để chạy toàn bộ pipeline trong notebook:
- Mở `Report_and_Demo.ipynb` trong Jupyter / JupyterLab.
- Nếu đã có `training_cache.pt`, notebook sẽ bỏ qua huấn luyện và tự động in lại các epoch đã lưu (replay logs).

## Ghi chú
- Nếu muốn luôn huấn luyện lại, xóa file `training_cache.pt` trước khi chạy notebook.
- Notebook có demo tương tác dùng `ipywidgets` để nhập lịch sử xem (autocomplete) và hiển thị phim gợi ý.

---
Nếu bạn muốn, tôi có thể tạo thêm `requirements.txt` hoặc bổ sung hướng dẫn môi trường ảo (venv/conda).