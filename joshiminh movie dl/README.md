# Deep Learning Movie Recommender System

A sophisticated next-item recommendation engine powered by sequence modeling (RNN, LSTM, GRU) and high-dimensional embeddings. This system predicts the next movie a user is likely to watch based on their historical watch sequence.

---

## 🚀 Key Features

*   **Sequence Modeling**: Uses Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU) to capture temporal dependencies in watch history.
*   **Dual Dataset Support**: Native integration for MovieLens 100K and MovieLens 1M datasets with a single configuration switch.
*   **Performance Dashboard**: A premium Streamlit UI to visualize training loss curves, compare model metrics (HitRate, NDCG), and test recommendations interactively.
*   **Flexible Training**: Interactive CLI supports model selection, optimizer tuning (Adam/SGD), and dynamic hardware device switching (CPU/CUDA).
*   **GPU Acceleration**: Full support for NVIDIA CUDA training for high-performance sequence modeling.

---

## 📂 Project Structure

```text
movie-recommender-dl/
├── artifacts/          # Trained model weights and performance metadata
├── config/             # YAML configuration for dataset, model, and training
├── data/               # MovieLens dataset storage
├── src/
│   ├── api/            # Inference service and FastAPI app
│   ├── data/           # Dataset loading and sequence preprocessing
│   ├── evaluation/     # Metrics (Hit@K, NDCG@K) and comparison tools
│   ├── models/         # PyTorch implementations of RNN, LSTM, and GRU
│   ├── training/       # Training loops and loss history tracking
│   ├── cli.py          # Interactive console interface
│   └── streamlit_app.py # Visualization and demo dashboard
├── main.py             # Application entry point
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

---

## 🛠️ Setup & Installation

### 1. Basic Installation (CPU)
```bash
pip install -r requirements.txt
```

### 2. GPU Support (Recommended for ML-1M)
If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch:

```bash
# Uninstall existing CPU version
pip uninstall torch torchvision torchaudio -y

# Install CUDA 12.4 version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### 3. Verify Hardware Detection
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

---

## 🎮 Usage Guide

### Launching the Console Menu
The primary entry point for managing the system:
```bash
python main.py
```
**Menu Options:**
1.  **Select Dataset**: Toggle between `ml-100k` and `ml-1m`.
2.  **Run Train**: Train a specific model or "all" models with your choice of optimizer and device.
3.  **Comprehensive Comparison**: Automatically trains all 6 combinations (RNN/LSTM/GRU x Adam/SGD).
4.  **Demo**: Quick command-line recommendation test.
5.  **Demo with UI**: Launches the Streamlit Performance Dashboard.

### Streamlit Dashboard
Visualize results and explore the neural engine:
```bash
streamlit run src/streamlit_app.py
```
*   **Recommender Studio**: Build custom watch sequences and see real-time predictions.
*   **Performance Dashboard**: Compare HitRate metrics and training loss curves across different model checkpoints.

---

## ⚙️ Configuration

### Dataset (`config/dataset.yaml`)
```yaml
dataset: ml-1m  # ml-100k or ml-1m
data_path: ./data/
```

### Model Architecture (`config/model.yaml`)
*   `embedding_dim`: Dimensionality of movie vectors.
*   `hidden_size`: Number of hidden units in the recurrent layer.
*   `max_seq_len`: Number of previous movies considered for each prediction.
*   `dropout`: Regularization factor to prevent overfitting.

---

## 📊 Performance & Optimization

The system compares different architectures and optimizers:
*   **Architectures**: LSTM and GRU typically outperform standard RNNs by effectively managing long-term memory.
*   **Optimizers**: Adam generally provides faster convergence and higher final HitRate compared to SGD.
*   **Metrics**: Evaluated using **Hit Rate @ K** (Did the actual next movie appear in the top K recommendations?) and **NDCG** (Was the actual next movie ranked highly?).

---

## 🆘 Troubleshooting

### CUDA "Illegal Instruction" Error
If training crashes on your GPU with an "illegal instruction" error:
1.  **Update Drivers**: Ensure your NVIDIA drivers are the latest version.
2.  **Switch to CPU**: During the CLI prompt, select **`cpu`** to bypass GPU hardware issues.
3.  **Use LSTM/GRU**: These models use more robust kernels than the basic RNN and are less likely to trigger hardware faults.

### Dataset Not Found
Ensure you have placed the MovieLens ZIP files (`ml-100k.zip` or `ml-1m.zip`) in the root or `data/` directory. The system will automatically extract them on the first run.

---

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.