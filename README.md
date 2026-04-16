# Acoustic Anomaly Detection

A decoupled, fullstack Machine Learning web application designed to automatically detect and classify anomalous urban sound events (such as sirens, gunshots, or car horns) in real-time.

---

## 📖 Overview
Computers struggle to find complex patterns in raw, chaotic 1D audio waves. This project bypasses that limitation by utilizing **Librosa** to convert raw audio into a 2D **Log-Mel Spectrogram** image representation. 

A custom-trained **PyTorch Convolutional Neural Network (CNN)**—built specifically to analyze these generated spectrogram images—then steps in to predict anomalies with extreme accuracy. Finally, a **FastAPI** Python backend seamlessly connects the model to a polished, custom **OLED Dark Mode UI**.

### Key Features
* 🧠 **Custom Deep Learning Model:** PyTorch CNN featuring Global Average Pooling.
* ⚡ **High-Speed API:** Asynchronous FastAPI backend capable of handling audio transformations on the fly.
* 🎨 **Bespoke Glassmorphic UI:** Raw Vanilla HTML, CSS, and JS. Zero bloat, perfectly stealth styling.
* ☁️ **Cloud Trained:** Accelerated via Nvidia T4 x2 GPUs on Kaggle using the UrbanSound8K dataset.

---

## 🛠️ Technology Stack
- **Deep Learning**: PyTorch, TorchAudio
- **Audio Processing**: Librosa
- **Backend API**: FastAPI, Uvicorn
- **Frontend / Dashboard**: HTML5, Vanilla JavaScript, CSS3
- **Data Engineering**: Pandas, Scikit-learn

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/raghav812/AcousticAnomalyDetection.git
cd AcousticAnomalyDetection
```

### 2. Environment Setup
It is highly recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Model Weights
Because the trained PyTorch `.pth` model binary files are large (often ~200MB+), they are excluded from this repository. 
- Ensure you train the model or download the pre-trained weights.
- Place the weights file here: `./checkpoints/acoustic_cnn_latest.pth`

### 4. Run the Fullstack Application
Since the frontend website is "mounted" perfectly into the FastAPI backend, you only need to run a single command to deploy the entire stack locally!

```bash
uvicorn api.main:app --reload
```

Then, open your web browser and navigate to:
`http://localhost:8000/`

---

## 📂 Project Structure
```text
AcousticAnomalyDetection/
├── api/
│   └── main.py              # Application Server & Inference Logic
├── checkpoints/             # Directory for .pth model weights
├── frontend/
│   ├── app.js               # Frontend fetch and animation logic
│   ├── index.html           # Structure of the dashboard
│   └── style.css            # OLED Glassmorphism Design System
├── src/
│   ├── data/
│   │   ├── dataset.py       # PyTorch DataLoader logic
│   │   └── generate_dummy_data.py
│   ├── models/
│   │   └── cnn.py           # Core Convolutional Neural Network Architecture
│   └── utils/
│       └── audio_utils.py   # Librosa Spectrogram math
├── requirements.txt         # Dependencies
└── README.md                # You are here!
```

---
*Built from the ground up for urban safety optimization and acoustic monitoring.*
