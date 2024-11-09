import click
import torch
from pathlib import Path
from .detector import BreathDetector
from .processor import AudioProcessor
import logging

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

@click.command()
@click.option('--input', '-i', required=True, help="Input audio file path")
@click.option('--output', '-o', required=True, help="Output audio file path")
@click.option('--model', '-m', help="Path to model file (optional, will download if not provided)")
@click.option('--sr', default=22050, help="Sample rate (default: 22050)")
@click.option('--channels', default=1, type=click.Choice(['1', '2']), 
              help="Number of channels (1=mono, 2=stereo)")
@click.option('--silence', default='full', callback=validate_silence,
              help="Silence level: 'full' or percentage 1-100 (default: full)")
@click.option('--plot', is_flag=True, help="Generate analysis plot")
def main(input, output, model, sr, channels, silence, plot):
    """Remove breath sounds from audio files."""
    try:
        # Check CUDA availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize detector and processor
        detector = BreathDetector(model_path=model, device=device)  # Now model_path can be None
        processor = AudioProcessor(
            detector=detector,
            sr=sr,
            channels=int(channels)
        )
               
        # Process file
        success = processor.process_file(
            input_file=input,
            output_file=output,
            silence_level=silence,
            plot=plot
        )
        
        if success:
            logger.info("Processing completed successfully!")
        else:
            logger.error("Processing failed!")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    main()