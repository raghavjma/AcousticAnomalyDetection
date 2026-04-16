# AcousticAnomalyDetection

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
