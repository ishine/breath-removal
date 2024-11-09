import click
import torch
from pathlib import Path
import logging
import numpy as np
import soundfile as sf
from .detector import BreathDetector
from .processor import AudioProcessor

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

def split_audio(audio, sr, max_minutes):
    """Split audio into segments if longer than max_minutes"""
    max_samples = int(max_minutes * 60 * sr)
    if len(audio) <= max_samples:
        return [audio]
    
    # Find silence points for better splitting
    from librosa.effects import split
    intervals = split(audio, top_db=30)
    
    segments = []
    current_segment = []
    current_length = 0
    
    for start, end in intervals:
        interval_audio = audio[start:end]
        if current_length + len(interval_audio) <= max_samples:
            current_segment.append(interval_audio)
            current_length += len(interval_audio)
        else:
            if current_segment:
                segments.append(np.concatenate(current_segment))
            current_segment = [interval_audio]
            current_length = len(interval_audio)
    
    if current_segment:
        segments.append(np.concatenate(current_segment))
    
    return segments

@click.command()
@click.option('--input', '-i', required=True, help="Input audio file path")
@click.option('--output', '-o', required=True, help="Output audio file path")
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
        # Check CUDA availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize detector
        detector = BreathDetector(model_path=model, device=device)
        
        # Load audio
        logger.info("Loading audio file...")
        audio, file_sr = sf.read(input)
        if file_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        
        # Convert to mono if needed
        if len(audio.shape) > 1 and channels == 1:
            audio = audio.mean(axis=1)
        elif len(audio.shape) == 1 and channels == 2:
            audio = np.stack([audio, audio])
        
        # Split audio if needed
        segments = split_audio(audio, sr, max_minutes)
        logger.info(f"Processing {len(segments)} segments...")
        
        # Process each segment
        processed_segments = []
        for i, segment in enumerate(segments):
            logger.info(f"Processing segment {i+1}/{len(segments)}...")
            processor = AudioProcessor(
                detector=detector,
                sr=sr,
                channels=channels
            )
            
            # Save temporary segment
            temp_file = f"temp_segment_{i}.wav"
            sf.write(temp_file, segment, sr)
            
            # Process segment
            output_file = f"temp_output_{i}.wav"
            success = processor.process_file(
                input_file=temp_file,
                output_file=output_file,
                silence_level=silence,
                plot=plot and i == 0  # Only plot first segment
            )
            
            if not success:
                logger.error("Processing failed!")
                return
                
            # Read processed segment
            processed_segment, _ = sf.read(output_file)
            processed_segments.append(processed_segment)
            
            # Clean up temporary files
            import os
            os.remove(temp_file)
            os.remove(output_file)
        
        # Concatenate and save final result
        final_audio = np.concatenate(processed_segments)
        sf.write(output, final_audio, sr)
        
        logger.info("Processing completed successfully!")
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    main()