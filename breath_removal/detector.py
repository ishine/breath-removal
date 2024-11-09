import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer

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
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model = DetectionNet().to(self.device)
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
        
        self.model.eval()