import torch
from torch import nn, Tensor
import warnings
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from metrics import SSIM, MS_SSIM


class MseLoss(nn.Module):
    def __init__(self,
                 smooth=False,
                 kernel_size=3,
                 in_channels=3,
                 device='cpu'
                 ):
        super(MseLoss, self).__init__()
        self.smooth = smooth
        self.device = device
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                              stride=1, padding=self.padding, bias=False)
        # 初始化卷积核为均值滤波权重
        kernel_weights = torch.ones(in_channels, in_channels, kernel_size, kernel_size) / (kernel_size ** 2)
        self.conv.weight.data = kernel_weights
        self.conv.to(device)
    
    def forward(self, inputs, targets, masks):
        if self.smooth:
            loss = self.l1(self.conv(inputs * masks), self.conv(targets * masks)) + self.mse(self.conv(inputs * (1 - masks)), self.conv(targets * (1 - masks)))
        else:            
            loss = self.l1(inputs * masks, targets * masks) + self.mse(inputs * (1 - masks), targets * (1 - masks))
        return loss


class SSIMLoss(MS_SSIM):
    def __init__(self,
                 data_range: float = 255,
                 size_average: bool = True,
                 win_size: int = 11,
                 win_sigma: float = 1.5,
                 channel: int = 3,
                 spatial_dims: int = 2,
                 weights: Optional[List[float]] = None,
                 K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
                 ):
        super(SSIMLoss, self).__init__(data_range=data_range,
                                       size_average=size_average,
                                       win_size=win_size,
                                       win_sigma=win_sigma,
                                       channel=channel,
                                       spatial_dims=spatial_dims,
                                       weights=weights,
                                       K=K
                                       )

    def forward(self, input, target):
        return 1 - super(SSIMLoss, self).forward(input*255, target*255)
