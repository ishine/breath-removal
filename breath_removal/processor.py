import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional
from intervaltree import Interval, IntervalTree
import logging
import torch

logger = logging.getLogger(__name__)

def feature_extractor(wav, sr=16000):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=int(sr*0.025), 
        hop_length=int(sr*0.01), n_mels=128
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    
    # ZCR extraction
    win_length = int(sr*0.025)
    hop_length = int(sr*0.01)
    pad_length = win_length // 2
    wav_padded = np.pad(wav, (pad_length, pad_length), 'constant')
    num_frames = 1 + (wav_padded.shape[0] - win_length) // hop_length
    zcrs = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        zcr = np.abs(np.sign(wav_padded[start+1:end]) - 
                    np.sign(wav_padded[start:end-1]))
        zcr = np.sum(zcr) * 0.5 / win_length
        zcrs[i] = zcr
        
    zcrs = zcrs.astype(np.float32)
    vms = np.var(mel, axis=0)

    mel = torch.tensor(mel).unsqueeze(0)
    zcr = torch.tensor(zcrs).unsqueeze(0)
    vms = torch.tensor(vms).unsqueeze(0)

    zcr = zcr.unsqueeze(1).expand(-1, 128, -1)
    vms = torch.var(mel, dim=1).unsqueeze(1).expand(-1, mel.shape[1], -1)

    feature = torch.stack((mel, vms, zcr), dim=1)
    length = torch.tensor([zcr.shape[-1]])
    return feature, length

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional
from intervaltree import Interval, IntervalTree
import logging
import torch

logger = logging.getLogger(__name__)

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
    """Split audio into segments at silence points."""
    max_samples = int(max_length * sr)
    if len(audio) <= max_samples:
        return [audio]
    
    segments = []
    start = 0
    
    while start < len(audio):
        end = min(start + max_samples, len(audio))
        segments.append(audio[start:end])
        start = end
    
    return segments

class AudioProcessor:
    def __init__(self, detector, sr=22050, channels=1):
        self.detector = detector
        self.sr = sr
        self.channels = channels

    def process_file(
        self,
        input_file: str,
        output_file: str,
        max_minutes: float = 5.0,
        sr: int = 22050,
        channels: int = 1,
        silence_level: Union[str, int] = 'full',
        plot: bool = False
    ) -> bool:
        try:
            # Load audio
            audio, file_sr = librosa.load(input_file, sr=sr, mono=(channels == 1))
            
            # Split audio into fixed-length segments
            max_length = max_minutes * 60  # Convert to seconds
            segments = split_audio(audio, sr, max_length)
            logger.info(f"Processing {len(segments)} segments...")
            
            processed_segments = []
            current_sample_offset = 0
            
            for i, segment in enumerate(segments):
                logger.info(f"Processing segment {i+1}/{len(segments)}...")
                segment_duration = len(segment) / sr
                
                # Get breath segments for this segment only
                breath_tree = self.detector.detect_breaths(segment, sr)
                
                # Convert tree to list of tuples with proper time offsets
                breath_segments = []
                for interval in sorted(breath_tree):
                    # Convert to local segment time
                    start = interval.begin
                    end = interval.end
                    if start < segment_duration and end > 0:  # Ensure segment is within bounds
                        breath_segments.append((start, end))
                
                # Process breaths for this segment
                processed_segment = self._silence_breaths(
                    segment,
                    breath_segments,
                    silence_level
                )
                
                if plot:
                    self._plot_analysis(
                        segment,
                        breath_segments,
                        str(Path(output_file).with_suffix('')) + f"_segment_{i+1}.png",
                        current_sample_offset / sr  # Convert samples to seconds for plotting
                    )
                
                processed_segments.append(processed_segment)
                current_sample_offset += len(segment)
            
            # Concatenate all processed segments
            final_audio = np.concatenate(processed_segments)
            sf.write(output_file, final_audio, sr)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in process_file: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _silence_breaths(
        self,
        audio: np.ndarray,
        breath_segments: List[Tuple[float, float]],
        silence_level: Union[str, int]
    ) -> np.ndarray:
        """Silence or reduce volume of breath segments."""
        processed_audio = audio.copy()
        reduction_factor = 0 if silence_level == 'full' else (1 - silence_level/100)
        
        for start, end in breath_segments:
            # Convert time to samples within this segment
            start_sample = int(start * self.sr)
            end_sample = int(end * self.sr)
            
            if start_sample >= end_sample or start_sample >= len(processed_audio):
                continue
                
            # Ensure we don't exceed segment boundaries
            end_sample = min(end_sample, len(processed_audio))
            
            # Calculate fade length (10ms or quarter of segment)
            segment_length = end_sample - start_sample
            fade_samples = min(int(0.01 * self.sr), segment_length // 4)
            
            if fade_samples > 0:
                # Create fade curves
                fade_out = np.linspace(1, reduction_factor, fade_samples)
                fade_in = np.linspace(reduction_factor, 1, fade_samples)
                
                # Apply fades
                processed_audio[start_sample:start_sample + fade_samples] *= fade_out
                
                if end_sample - fade_samples > start_sample + fade_samples:
                    processed_audio[start_sample + fade_samples:end_sample - fade_samples] *= reduction_factor
                
                processed_audio[end_sample - fade_samples:end_sample] *= fade_in
                
                logger.debug(f"Processed breath at {start:.3f}s-{end:.3f}s")
            else:
                processed_audio[start_sample:end_sample] *= reduction_factor
        
        return processed_audio
            
    def _plot_analysis(
        self,
        audio: np.ndarray,
        breath_segments: List[Tuple[float, float]],
        output_path: str,
        segment_offset: float = 0.0
    ) -> None:
        """Generate visualization of audio with breath markers"""
        plt.figure(figsize=(15, 5))
        
        # Calculate time array with offset
        duration = len(audio) / self.sr
        time = np.linspace(segment_offset, segment_offset + duration, len(audio))
        
        # Plot waveform
        plt.plot(time, audio, color='blue', alpha=0.6, linewidth=1)
        
        # Plot breath segments
        for start, end in breath_segments:
            # Only plot segments that fall within this segment's time range
            if start >= segment_offset and start < (segment_offset + duration):
                plt.axvspan(start, end, color='red', alpha=0.3, label='Breath')
        
        plt.title(f"Waveform Analysis ({segment_offset:.1f}s - {segment_offset + duration:.1f}s)")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Add legend if there are breath segments
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
        
        # Set x-axis limits
        plt.xlim(segment_offset, segment_offset + duration)
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved analysis plot to {output_path}")