import click
import torch
import numpy as np
from pathlib import Path
import logging
from typing import List
import soundfile as sf
import librosa
from .detector import BreathDetector
from .processor import AudioProcessor
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def validate_silence(ctx, param, value):
    if value.lower() == 'full':
        return 'full'
    try:
        val = int(value)
        if 1 <= val <= 100:
            return val
        raise ValueError
    except ValueError:
        raise click.BadParameter("Silence must be 'full' or an integer between 1 and 100")

def find_silence_points(audio: np.ndarray, sr: int, min_silence_len: float = 0.3, silence_thresh: float = -40) -> List[int]:
    """Find silence points in audio that can be used as split points"""
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Calculate frame length for minimum silence length
    frame_length = int(sr * min_silence_len)
    hop_length = frame_length // 4
    
    # Calculate RMS energy for each frame
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert to dB
    db = librosa.amplitude_to_db(rms)
    
    # Find silence frames
    silence_frames = np.where(db < silence_thresh)[0]
    
    # Convert frame indices to sample indices
    silence_points = silence_frames * hop_length
    
    # Remove points that are too close to the start or end
    silence_points = silence_points[
        (silence_points > sr) & (silence_points < len(audio) - sr)
    ]
    
    return sorted(list(silence_points))

def split_audio(audio: np.ndarray, sr: int, max_length: float) -> List[np.ndarray]:
    """Split audio into segments at silence points"""
    if len(audio) / sr <= max_length:
        return [audio]
    
    max_samples = int(max_length * sr)
    silence_points = find_silence_points(audio, sr)
    
    if not silence_points:
        # If no silence points found, split at max_length
        return np.array_split(audio, np.ceil(len(audio) / max_samples))
    
    segments = []
    start = 0
    
    while start < len(audio):
        # Find the best silence point within max_length
        end_target = start + max_samples
        suitable_points = [p for p in silence_points if start < p <= end_target]
        
        if suitable_points:
            end = suitable_points[-1]
        else:
            end = min(end_target, len(audio))
        
        segments.append(audio[start:end])
        start = end
    
    return segments

@click.command()
@click.option('--input', '-i', required=True, help="Input audio file path")
@click.option('--output', '-o', required=True, help="Output folder path")
@click.option('--model', '-m', help="Path to model file (optional, will download if not provided)")
@click.option('--sr', default=22050, help="Sample rate (default: 22050)")
@click.option('--channels', type=int, default=1, help="Number of channels (default: 1)")
@click.option('--silence', default='full', callback=validate_silence,
              help="Silence level: 'full' or percentage 1-100 (default: full)")
@click.option('--plot', is_flag=True, help="Generate analysis plot")
@click.option('--max-minutes', type=float, default=5.0, 
              help="Maximum segment length in minutes (default: 5.0)")
def main(input, output, model, sr, channels, silence, plot, max_minutes):
    """Remove breath sounds from audio files."""
    try:
        # Create output folder if needed
        os.makedirs(output, exist_ok=True)
        
        # Check CUDA availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize detector
        detector = BreathDetector(model_path=model, device=device)
        
        # Load audio
        logger.info("Loading audio file...")
        audio, file_sr = librosa.load(input, sr=sr, mono=(channels == 1))
        logger.info(f"Loaded audio shape: {audio.shape}, sr: {file_sr}")
        
        # Generate output filename
        input_name = os.path.basename(input)
        output_file = os.path.join(output, f"breath_removal_{input_name}")
        
        # Process audio
        processor = AudioProcessor(detector=detector)
        success = processor.process_file(
            input_file=input,
            output_file=output_file,
            max_minutes=max_minutes,
            sr=sr,
            channels=channels,
            silence_level=silence,
            plot=plot
        )
        
        if success:
            logger.info("Processing completed successfully!")
        else:
            logger.error("Processing failed!")
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise click.ClickException(str(e))