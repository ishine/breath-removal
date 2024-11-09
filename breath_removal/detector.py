import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer
from intervaltree import Interval, IntervalTree
import os
from pathlib import Path
import logging
import requests
from tqdm import tqdm
import tempfile

logger = logging.getLogger(__name__)

MODEL_URL = "https://huggingface.co/DongYANG/Respiro-en/resolve/main/respiro-en.pt"
MODELS_DIR = Path.home() / ".breath_removal" / "models"

def download_model(force: bool = False) -> Path:
    """Download the Respiro-en model if not present."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "respiro-en.pt"
    
    if model_path.exists() and not force:
        logger.info(f"Model already exists at {model_path}")
        return model_path
        
    logger.info("Downloading Respiro-en model...")
    
    # Stream download with progress bar
    response = requests.get(MODEL_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(model_path, 'wb') as f, tqdm(
        desc="Downloading model",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
            
    logger.info(f"Model downloaded to {model_path}")
    return model_path

def zcr_extractor(wav, win_length, hop_length):
    """Extract zero-crossing rate feature."""
    pad_length = win_length // 2
    wav = np.pad(wav, (pad_length, pad_length), 'constant')
    num_frames = 1 + (wav.shape[0] - win_length) // hop_length
    zcrs = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        zcr = np.abs(np.sign(wav[start+1:end]) - np.sign(wav[start:end-1]))
        zcr = np.sum(zcr) * 0.5 / win_length
        zcrs[i] = zcr
        
    return zcrs.astype(np.float32)

def feature_extractor(wav, sr=16000):
    """Extract mel spectrogram, ZCR and variance features."""
    # Ensure audio is float32 and normalized
    wav = wav.astype(np.float32)
    if wav.max() > 1.0 or wav.min() < -1.0:
        wav = wav / max(abs(wav.max()), abs(wav.min()))
    
    # Extract mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, 
        n_fft=int(sr*0.025), 
        hop_length=int(sr*0.01), 
        n_mels=128
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    
    # Extract ZCR
    zcr = zcr_extractor(
        wav, 
        win_length=int(sr*0.025), 
        hop_length=int(sr*0.01)
    )
    
    # Calculate variance
    vms = np.var(mel, axis=0)

    # Convert to tensors
    mel = torch.tensor(mel).unsqueeze(0)
    zcr = torch.tensor(zcr).unsqueeze(0)
    vms = torch.tensor(vms).unsqueeze(0)

    # Match dimensions
    zcr = zcr.unsqueeze(1).expand(-1, 128, -1)
    vms = torch.var(mel, dim=1).unsqueeze(1).expand(-1, mel.shape[1], -1)

    # Stack features
    feature = torch.stack((mel, vms, zcr), dim=1)
    length = torch.tensor([zcr.shape[-1]])
    
    return feature, length

class Conv2dDownsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dDownsampling, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
    def forward(self, x, length):
        keep_dim_padding = 1 - x.shape[-1] % 2
        x = F.pad(x, (0, keep_dim_padding, 0, 0))
        x = self.conv1(x)
        length = (length - 3 + keep_dim_padding) // 2 + 1
        
        keep_dim_padding = 1 - x.shape[-1] % 2
        x = F.pad(x, (0, keep_dim_padding, 0, 0))
        x = self.conv2(x)
        length = (length - 3 + keep_dim_padding) // 2 + 1
        return x, length

class Conv1dUpsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv1dUpsampling, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.deconv(x)
        return x

class DetectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsampling = Conv2dDownsampling(3, 1)
        self.upsampling = Conv1dUpsampling(128, 128)
        self.linear = nn.Linear(31, 128)
        self.dropout = nn.Dropout(0.1)
        
        self.conformer = Conformer(
            input_dim=128,
            num_heads=4,
            ffn_dim=256,
            num_layers=8,
            depthwise_conv_kernel_size=31,
            dropout=0.1
        )
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            bidirectional=True,
            batch_first=True
        )
        
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        sequence = x.shape[-1]
        x, length = self.downsampling(x, length)
        x = x.squeeze(1).transpose(1, 2).contiguous()
        x = self.linear(x)
        x = self.dropout(x)
        x = self.conformer(x, length)[0]
        x = x.transpose(1, 2).contiguous()
        x = self.upsampling(x)
        x = x.transpose(1, 2).contiguous()
        x = self.lstm(x)[0]
        x = self.fc(x)
        x = self.sigmoid(x.squeeze(-1))
        return x[:, :sequence]

class BreathDetector:
    def __init__(self, model_path=None, device=None):
        """Initialize breath detector with optional model path."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model = DetectionNet().to(self.device)
        
        if model_path is None:
            model_path = download_model()
            
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def detect_breaths(self, audio: np.ndarray, sr: int, threshold: float = 0.064, min_length: int = 20) -> IntervalTree:
        """Detect breath segments in audio array."""
        # Create temporary 16kHz version for detection
        if sr != 16000:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                # Save temporary file at 16kHz
                temp_audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                feature, length = feature_extractor(temp_audio, sr=16000)
        else:
            feature, length = feature_extractor(audio, sr=16000)
            
        feature, length = feature.to(self.device), length.to(self.device)
        
        with torch.no_grad():
            output = self.model(feature, length)
        
        # Process predictions
        prediction = (output[0] > threshold).nonzero().squeeze().tolist()
        tree = IntervalTree()
        
        if isinstance(prediction, list) and len(prediction) > 1:
            diffs = np.diff(prediction)
            splits = np.where(diffs != 1)[0] + 1
            splits = np.split(prediction, splits)
            splits = list(filter(lambda split: len(split) > min_length, splits))
            
            # Convert frame indices to time and create intervals
            for split in splits:
                if len(split) > 0:
                    # Convert frame indices to seconds, adjusting for original sample rate
                    start_time = split[0] * 0.01 * (sr / 16000)
                    end_time = split[-1] * 0.01 * (sr / 16000)
                    if end_time > start_time:
                        tree.add(Interval(
                            round(start_time, 2),
                            round(end_time, 2)
                        ))
        
        return tree

    def __call__(self, wav_path: str, threshold: float = 0.064, min_length: int = 20) -> IntervalTree:
        """Detect breaths in audio file."""
        # Load audio at its original sample rate
        audio, sr = librosa.load(wav_path, sr=None)
        return self.detect_breaths(audio, sr, threshold, min_length)