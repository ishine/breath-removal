import click
import torch
import os
import logging
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
        
        # Generate output filename
        input_name = os.path.basename(input)
        output_file = os.path.join(output, f"breath_removal_{input_name}")
        
        # Process audio
        processor = AudioProcessor(detector=detector, sr=sr, channels=channels)
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

if __name__ == '__main__':
    main()