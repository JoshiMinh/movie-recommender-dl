# So sánh Yêu cầu Project vs Hiện trạng Code

## Tổng quan

Project 5: **Hệ thống khuyến nghị phim sử dụng Deep Learning (Embedding + RNN)** — Next-item recommendation trên MovieLens.

---

## Bảng so sánh chi tiết

| # | Yêu cầu | Trạng thái | Ghi chú |
|---|---------|:----------:|---------|
| **1. Mục tiêu** | Dự đoán phim người dùng sẽ thích (next-item) | ✅ | Đã implement đúng next-item classification |
| **2. Dataset** | MovieLens 100K hoặc 1M (`user_id`, `movie_id`, `rating`, `timestamp`) | ✅ | Dùng **ML-1M** trong [dataset.py](file:///d:/HOC%20TAP/HK%206%20%282025-2026%29/Deep%20Learning/MiniProject/dataset.py) |
| **3.1** | Embedding layer: user embedding, movie embedding | ✅ | [model.py](file:///d:/HOC%20TAP/HK%206%20%282025-2026%29/Deep%20Learning/MiniProject/model.py) L11-12: `nn.Embedding` cho cả user và movie |
| **3.2** | Sequence modeling: RNN / LSTM / GRU | ✅ | [model.py](file:///d:/HOC%20TAP/HK%206%20%282025-2026%29/Deep%20Learning/MiniProject/model.py) L18-26: Configurable `rnn_type` |
| **3.3** | Output layer: Softmax trên tất cả movie | ✅ | [model.py](file:///d:/HOC%20TAP/HK%206%20%282025-2026%29/Deep%20Learning/MiniProject/model.py) L31: `nn.Linear(hidden_dim, num_movies)` + [predict()](file:///d:/HOC%20TAP/HK%206%20%282025-2026%29/Deep%20Learning/MiniProject/model.py#70-74) dùng softmax |
| **3.4** | Pipeline: User sequence → Embedding → RNN → Dense → Softmax | ✅ | Đúng pipeline trong [forward()](file:///d:/HOC%20TAP/HK%206%20%282025-2026%29/Deep%20Learning/MiniProject/model.py#33-69) và [predict()](file:///d:/HOC%20TAP/HK%206%20%282025-2026%29/Deep%20Learning/MiniProject/model.py#70-74) |
| **4.1** | Model base: Embedding + RNN | ✅ | |
| **4.2** | So sánh model: RNN vs LSTM vs GRU (**bảng so sánh**) | ✅ | Notebook section 2: loop qua 3 architectures + hiển thị bảng `df_arch` |
| **4.3** | Learning Rate tuning | ✅ | Notebook section 3: test `[1e-2, 1e-3, 1e-4]` |
| **4.4** | Dropout | ✅ | [model.py](file:///d:/HOC%20TAP/HK%206%20%282025-2026%29/Deep%20Learning/MiniProject/model.py) L30: `nn.Dropout(dropout)` với default `0.3` |
| **4.5** | Optimizer: SGD; Adam | ✅ | Notebook section 3: so sánh `['sgd', 'adam']` |
| **4.6** | Metrics | ✅ | [evaluate.py](file:///d:/HOC%20TAP/HK%206%20%282025-2026%29/Deep%20Learning/MiniProject/evaluate.py): **Top-1 Accuracy** + **Hit Ratio @10** |
| **5.1** | Demo: User đã xem [Titanic, Avatar, Inception] → Gợi ý | ✅ | Notebook section 4: `predict_next_movie(["Titanic", "Avatar", "Inception"])` |
| **5.2** | Bảng so sánh LSTM vs GRU vs RNN | ✅ | Notebook: `df_arch` + `df_opt` |
| **5.3** | Visualization: Loss curve | ✅ | Notebook: `plt.plot()` cho cả architecture comparison và optimizer tuning |

---

## Kết luận

> [!TIP]
> **Project đã đáp ứng đầy đủ tất cả các yêu cầu** từ đề bài trong ảnh.

### Checklist tổng hợp

- ✅ Dataset MovieLens 1M
- ✅ Embedding layer (user + movie) 
- ✅ Sequence modeling với RNN / LSTM / GRU configurable
- ✅ Pipeline đúng: Embedding → RNN → Dense → Softmax
- ✅ So sánh 3 kiến trúc (bảng so sánh)
- ✅ Optimization: Learning Rate tuning, Dropout, SGD vs Adam
- ✅ Metrics (Top-1 Accuracy, HR@10)
- ✅ Demo interactive (input titles → recommendation)
- ✅ Loss curve visualization
- ✅ OOM fallback (batch size tự giảm)
- ✅ Reproducibility (seed = 42)
- ✅ Per-user chronological split (80/10/10)

### Điểm mạnh bổ sung (vượt yêu cầu)

| Tính năng | Chi tiết |
|-----------|----------|
| OOM fallback | [train.py](file:///d:/HOC%20TAP/HK%206%20%282025-2026%29/Deep%20Learning/MiniProject/train.py): Tự động giảm batch size khi CUDA OOM |
| Packed sequences | [model.py](file:///d:/HOC%20TAP/HK%206%20%282025-2026%29/Deep%20Learning/MiniProject/model.py): Xử lý padding chính xác với `pack_padded_sequence` |
| Last valid hidden state | Trích đúng hidden state cuối cùng (không lấy padding) |
| Modular codebase | Tách rõ 4 file [.py](file:///d:/HOC%20TAP/HK%206%20%282025-2026%29/Deep%20Learning/MiniProject/train.py) + 1 notebook orchestrator |
| [project_specs.md](file:///d:/HOC%20TAP/HK%206%20%282025-2026%29/Deep%20Learning/MiniProject/project_specs.md) | Có spec chi tiết 327 dòng |

