import os
import logging
from pathlib import Path
import torch
import requests
from tqdm import tqdm

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

def get_model_path() -> Path:
    """Get path to model, downloading if necessary."""
    return download_model()