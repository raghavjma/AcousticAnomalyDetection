import librosa
import numpy as np

# Global Audio Parameters
SAMPLE_RATE = 22050
DURATION = 4.0 # in seconds
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

def load_and_preprocess_audio(file_path, sample_rate=SAMPLE_RATE, duration=DURATION):
    """
    Load an audio file, pad or truncate it to the target duration, 
    and compute its log mel-spectrogram.
    """
    # Load audio
    y, sr = librosa.load(file_path, sr=sample_rate)
    
    # Target length in samples
    target_length = int(sample_rate * duration)
    
    # Pad or truncate
    if len(y) > target_length:
        y = y[:target_length]
    else:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), mode='constant')
        
    # Convert to log mel-spectrogram
    # The output is of shape (n_mels, time_steps)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Expand dimensions to add the 'channel' axis for the CNN (1, n_mels, time_steps)
    log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=0)
    
    return log_mel_spectrogram
