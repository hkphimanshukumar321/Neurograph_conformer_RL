import torch
import torch.nn as nn
import torch.nn.functional as F

class FilterBankFrontEnd(nn.Module):
    """
    Learnable FilterBank Front-End for EEG.
    Replaces static Morlet CWT with a fast 1D Depthwise Convolutional filter bank
    to extract time-frequency features on the GPU.
    """
    def __init__(self, in_channels: int, num_filters: int = 16, kernel_size: int = 64, stride: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        
        # Depthwise convolution: groups=in_channels ensures each EEG channel is filtered independently.
        # Output channels = in_channels * num_filters
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * num_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )
        self.bn = nn.BatchNorm1d(in_channels * num_filters)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw EEG signal
        Returns:
            out: (B, C, F, T') time-frequency features
        """
        B, C, T = x.shape
        
        # (B, C * F, T')
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        
        # Reshape to explicitly separate channels and frequency bands
        T_prime = out.shape[-1]
        out = out.view(B, C, self.num_filters, T_prime)
        
        return out
