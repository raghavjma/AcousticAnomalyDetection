import os
import sys
import base64
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn.functional as F
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Adjust python path so we can import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.cnn import AcousticCNN
from src.utils.audio_utils import load_and_preprocess_audio, SAMPLE_RATE

app = FastAPI(title="Acoustic Anomaly API")

# Setup CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = os.path.join(os.path.dirname(__file__), '../checkpoints/acoustic_cnn_latest.pth')
model = AcousticCNN(num_classes=2)

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("✅ Model weights loaded successfully.")
else:
    print("⚠️ No weights found at checkpoints/acoustic_cnn_latest.pth. Using random initialization.")

model.to(device)
model.eval()


def generate_spectrogram_base64(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1E293B')
    ax.axis('off') # Hide axes for a cleaner UI look
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    plt.close(fig)
    
    # Encode to base64
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{base64_img}"


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")
        
    temp_file_path = f"temp_{file.filename}"
    try:
        # Save temp file
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # 1. Model Inference
        input_tensor_numpy = load_and_preprocess_audio(temp_file_path)
        input_tensor = torch.tensor(input_tensor_numpy, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        normal_prob = float(probabilities[0]) * 100
        anomaly_prob = float(probabilities[1]) * 100
        
        # 2. Generate Spectrogram Graphic
        b64_image = generate_spectrogram_base64(temp_file_path)
        
        return JSONResponse(content={
            "success": True,
            "normal_probability": normal_prob,
            "anomaly_probability": anomaly_prob,
            "is_anomaly": anomaly_prob > 50,
            "spectrogram_b64": b64_image,
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Mount the frontend directory to serve the UI
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), '../frontend'), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
