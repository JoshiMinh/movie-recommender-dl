# Movie Recommendation System - Deep Learning Edition

## Overview

A state-of-the-art next-item movie recommendation engine powered by sequence modeling using Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU). This system predicts the next movie a user will watch based on their historical viewing sequence, utilizing high-dimensional embeddings and temporal dependency learning.

## 🚀 Key Features

- **Sequential Recommendation**: Captures temporal dependencies in user watch history using RNN architectures
- **Multiple Model Architectures**: Compare RNN, LSTM, and GRU implementations with unified interface
- **Flexible Dataset Support**: Native integration for MovieLens 100K and MovieLens 1M datasets
- **Comprehensive Training Pipeline**: Interactive CLI with optimizer selection (Adam/SGD), dynamic learning rate tuning, and automatic GPU acceleration
- **Interactive Visualization**: Streamlit-based performance dashboard with real-time metric tracking and recommendation testing
- **Production-Ready API**: FastAPI inference service for integration into production systems
- **Robust Error Handling**: Automatic out-of-memory fallback with batch size optimization

## 📂 Project Structure

```
movie-recommender-dl/
├── main.py                 # Application entry point
├── Movie_Recommendation_Pipeline.ipynb   # Complete pipeline demonstration and analysis
├── src/
│   ├── app.py              # FastAPI inference service
│   ├── dataset.py          # Dataset handling utilities
│   ├── inference.py        # Recommender service wrapper
│   ├── metrics.py          # Evaluation metrics
│   ├── model.py            # Model architecture definitions
│   ├── streamlit.py        # Dashboard UI
│   ├── train.py            # Training logic
│   └── utils.py            # Shared utilities
├── data/                   # MovieLens dataset storage
├── requirements.txt        # Python dependencies
└── LICENSE                 # MIT License
```

## 🛠️ Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd movie-recommender-dl
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. GPU Support (Optional but Recommended)
For NVIDIA GPU acceleration with CUDA:
```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify Installation
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

## 🎮 Quick Start Guide

### Interactive Console Application
```bash
python main.py
```

**Menu Options:**
1. Select dataset (ml-100k or ml-1m)
2. Train specific model or all architectures
3. Run comprehensive comparison (all model × optimizer combinations)
4. Quick CLI recommendation test
5. Launch Streamlit interactive dashboard

### Streamlit Performance Dashboard
```bash
streamlit run src/streamlit.py
```

Features:
- **Recommender Studio**: Build custom watch sequences for real-time predictions
- **Performance Comparison**: Visualize metrics across different model checkpoints
- **Training Curves**: Monitor loss convergence during training

### FastAPI Inference Service
```bash
uvicorn src.app:app --reload
```

## ⚙️ Configuration

### Dataset Configuration (`config/dataset.yaml`)
```yaml
dataset: ml-1m        # Options: ml-100k, ml-1m
data_path: ./data/
max_seq_len: 50       # Maximum sequence length for recommendations
```

### Model Configuration (`config/model.yaml`)
```yaml
model: lstm           # Options: rnn, lstm, gru
embedding_dim: 64
hidden_size: 128
dropout: 0.3
```

### Training Configuration (`config/train.yaml`)
```yaml
epochs: 20
batch_size: 256
learning_rate: 0.001
optimizer: adam       # Options: adam, sgd
```

## 📊 Technical Details

### Architecture
- **Embedding Layer**: Learns dense representations for movies and users
- **Recurrent Layer**: Captures sequential dependencies (RNN/LSTM/GRU)
- **Output Layer**: Softmax classification over all movies

### Data Preprocessing
- Per-user chronological split (80% train, 10% validation, 10% test)
- Sliding window sequences with maximum length of 50 items
- Automatic leakage prevention with split-specific sequence generation
- Padding strategy: left-padding with index 0

### Evaluation Metrics
- **Top-1 Accuracy**: Percentage of correct next-item predictions
- **Hit Ratio@10**: Percentage of target items in top-10 predictions
- **NDCG@K**: Normalized Discounted Cumulative Gain for ranking quality

## 🔍 Usage Examples

### Basic Recommendation
```python
from src.inference import RecommenderService

service = RecommenderService.from_default()
recommendations = service.recommend([101, 102, 103], top_k=5)
```

### Custom Model Training
```python
from src.dataset import MovieSequenceDataset, process_data, load_data
from src.model import SequenceRecommender
from src.trainer import train_with_oom_fallback

ratings_df, movies_df = load_data()
train_data, val_data, test_data, movie2idx, user2idx = process_data(ratings_df)

# Train with automatic OOM handling
model, train_losses, val_losses, batch_size = train_with_oom_fallback(
    create_model_fn=lambda: SequenceRecommender(
        num_users=len(user2idx),
        num_movies=len(movie2idx),
        rnn_type='lstm',
        hidden_dim=128
    ),
    train_dataset=MovieSequenceDataset(train_data),
    val_dataset=MovieSequenceDataset(val_data),
    optimizer_name='adam',
    lr=1e-3,
    num_epochs=20
)
```

## 📈 Performance Benchmarks

The system achieves competitive results across different architectures:
- **LSTM**: Superior performance with ~25-30% Top-1 Accuracy on MovieLens 1M
- **GRU**: Faster training with comparable performance
- **RNN**: Baseline model for architecture comparison

Results vary based on dataset size, sequence length, and hyperparameter tuning.

## 🔐 Reproducibility

All experiments use fixed random seeds for reproducibility:
```python
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

## 📝 Citation

If you use this system in your research, please cite:
```
@software{movie_recommender_2026,
  authors = {JoshiMinh, Jade2308},
  title = {Movie Recommendation System - Deep Learning Edition},
  year = {2026},
  url = {https://github.com/JoshiMinh/movie-recommender-dl}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Authors**: JoshiMinh, Jade2308

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Support

For questions or support, please open an issue on the GitHub repository or contact the project maintainers.

---

**Last Updated**: May 2, 2026