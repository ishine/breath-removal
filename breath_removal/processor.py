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

class AudioProcessor:
    def __init__(self, detector, sr=22050, channels=1):
        self.detector = detector
        self.sr = sr
        self.channels = channels
        
    def process_file(
        self,
        input_file: str,
        output_file: str,
        silence_level: Union[str, int] = 'full',
        plot: bool = False
    ) -> bool:
        try:
            # Load audio
            audio, _ = librosa.load(input_file, sr=self.sr, mono=(self.channels == 1))
            
            # Detect breaths
            features, length = feature_extractor(audio)
            features = features.to(self.detector.device)
            length = length.to(self.detector.device)
            
            with torch.no_grad():
                output = self.detector.model(features, length)
            
            # Convert to breath segments
            threshold = 0.064  # from original repo
            prediction = (output[0] > threshold).nonzero().cpu().numpy()
            
            breath_segments = []
            if len(prediction) > 0:
                # Group consecutive frames into segments
                breaks = np.where(np.diff(prediction[:, 0]) > 1)[0] + 1
                segments = np.split(prediction[:, 0], breaks)
                
                for segment in segments:
                    if len(segment) > 20:  # minimum length threshold
                        start = segment[0] * 0.01  # convert frame index to time
                        end = segment[-1] * 0.01
                        breath_segments.append((start, end))
            
            # Process breaths
            if breath_segments:
                processed_audio = self._silence_breaths(
                    audio, breath_segments, silence_level
                )
            else:
                processed_audio = audio
            
            # Generate plot if requested
            if plot:
                self._plot_analysis(audio, breath_segments, output_file)
            
            # Save processed audio
            sf.write(output_file, processed_audio, self.sr)
            return True
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return False
    
    def _silence_breaths(
        self,
        audio: np.ndarray,
        breath_segments: List[Tuple[float, float]],
        silence_level: Union[str, int]
    ) -> np.ndarray:
        processed_audio = audio.copy()
        reduction_factor = 0 if silence_level == 'full' else (1 - silence_level/100)
        
        for start, end in breath_segments:
            start_sample = int(start * self.sr)
            end_sample = int(end * self.sr)
            
            if start_sample >= end_sample or start_sample >= len(processed_audio):
                continue
                
            fade_samples = min(int(0.01 * self.sr), (end_sample - start_sample) // 4)
            
            if fade_samples > 0:
                fade_out = np.linspace(1, reduction_factor, fade_samples)
                fade_in = np.linspace(reduction_factor, 1, fade_samples)
                
                processed_audio[start_sample:start_sample + fade_samples] *= fade_out
                
                if start_sample + fade_samples < end_sample - fade_samples:
                    processed_audio[start_sample + fade_samples:end_sample - fade_samples] *= reduction_factor
                
                if end_sample - fade_samples > start_sample:
                    processed_audio[end_sample - fade_samples:end_sample] *= fade_in
            else:
                processed_audio[start_sample:end_sample] *= reduction_factor
                
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