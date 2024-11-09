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
            
            # Split audio if needed
            max_length = max_minutes * 60  # Convert to seconds
            segments = split_audio(audio, sr, max_length)
            logger.info(f"Processing {len(segments)} segments...")
            
            processed_segments = []
            for i, segment in enumerate(segments):
                logger.info(f"Processing segment {i+1}/{len(segments)}...")
                
                # Process breath detection on segment
                features, length = feature_extractor(segment)
                features = features.to(self.detector.device)
                length = length.to(self.detector.device)
                
                with torch.no_grad():
                    output = self.detector.model(features, length)
                
                # Process breaths in segment
                threshold = 0.064
                prediction = (output[0] > threshold).nonzero().cpu().numpy()
                
                if len(prediction) > 0:
                    # Group consecutive frames
                    breaks = np.where(np.diff(prediction[:, 0]) > 1)[0] + 1
                    breath_segments = np.split(prediction[:, 0], breaks)
                    
                    # Convert to time segments
                    breath_times = []
                    for breath_segment in breath_segments:
                        if len(breath_segment) > 20:  # minimum length threshold
                            start = breath_segment[0] * 0.01
                            end = breath_segment[-1] * 0.01
                            breath_times.append((start, end))
                            
                    # Process breaths
                    segment = self._silence_breaths(segment, breath_times, silence_level)
                    
                processed_segments.append(segment)
                
            # Concatenate and save
            final_audio = np.concatenate(processed_segments)
            sf.write(output_file, final_audio, sr)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in process_file: {str(e)}")
            return False

    def _silence_breaths(
        self,
        audio: np.ndarray,
        breath_segments: List[Tuple[float, float]],
        silence_level: Union[str, int]
    ) -> np.ndarray:
        """Silence or reduce volume of breath segments"""
        processed_audio = audio.copy()
        reduction_factor = 0 if silence_level == 'full' else (1 - silence_level/100)
        
        # Skip processing if no valid breath segments
        if not breath_segments:
            return processed_audio
            
        for start, end in breath_segments:
            # Convert time to samples
            start_sample = max(0, int(start * self.sr))
            end_sample = min(len(processed_audio), int(end * self.sr))
            
            # Skip if segment is invalid
            if start_sample >= end_sample or start_sample >= len(processed_audio):
                continue
                
            # Calculate fade length based on segment duration
            segment_length = end_sample - start_sample
            fade_samples = min(int(0.01 * self.sr), segment_length // 4)
            
            # Skip if segment is too short for processing
            if segment_length <= 2:
                continue
                
            try:
                if fade_samples > 0:
                    # Create fade curves
                    fade_out = np.linspace(1, reduction_factor, fade_samples)
                    fade_in = np.linspace(reduction_factor, 1, fade_samples)
                    
                    # Apply fade out
                    processed_audio[start_sample:start_sample + fade_samples] *= fade_out
                    
                    # Apply reduction to middle part if there is one
                    if start_sample + fade_samples < end_sample - fade_samples:
                        processed_audio[start_sample + fade_samples:end_sample - fade_samples] *= reduction_factor
                    
                    # Apply fade in
                    if end_sample - fade_samples > start_sample:
                        processed_audio[end_sample - fade_samples:end_sample] *= fade_in
                else:
                    # For very short segments, simple reduction
                    processed_audio[start_sample:end_sample] *= reduction_factor
                    
            except ValueError as e:
                logger.warning(f"Skipping problematic breath segment {start:.2f}s-{end:.2f}s: {str(e)}")
                continue
        
        return processed_audio
    
    def _plot_analysis(
        self,
        audio: np.ndarray,
        breath_segments: List[Tuple[float, float]],
        output_path: str
    ) -> None:
        plt.figure(figsize=(15, 5))
        
        time = np.linspace(0, len(audio)/self.sr, len(audio))
        plt.plot(time, audio, color='blue', alpha=0.6, linewidth=1)
        
        for start, end in breath_segments:
            plt.axvspan(start, end, color='red', alpha=0.3, label='Breath')
        
        plt.title("Waveform Analysis with Breath Segments")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        if breath_segments:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
        
        plot_path = str(Path(output_path).with_suffix('')) + "_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()