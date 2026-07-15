import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBurstModel(nn.Module):
    """
    A simple convolutional burst model.
    Input: [B, 9, C, H, W] (9 burst frames)
    Output: [B, 3, H, W] (restored image)
    """

    def __init__(self, in_frames=9, in_channels=3, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_frames * in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.reshape(B, N * C, H, W)  # merge burst frames into channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x