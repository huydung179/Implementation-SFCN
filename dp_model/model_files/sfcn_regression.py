import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class SFCNRegression(nn.Module):
    def __init__(self, dropout: float = .0, weight_decay: float = .0,
        include_top: bool = True, depths: List[int] = [32, 64, 128, 256, 256, 64],
        prediction_range: Tuple[float, float] = (0, 95), weights: str = None):
        super(SFCNRegression, self).__init__()
        self.prediction_range = prediction_range
        self.include_top = include_top

        n_layer = len(depths)
        self.feature_extractor = nn.Sequential()

        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = depths[i-1]
            out_channel = depths[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    f'conv_{i}',
                    self.conv_layer(
                        in_channel,
                        out_channel,
                        maxpool=True,
                        kernel_size=3,
                        padding='same'
                    )
                )
            else:
                self.feature_extractor.add_module(
                    f'conv_{i}',
                    self.conv_layer(
                        in_channel,
                        out_channel,
                        maxpool=False,
                        kernel_size=1,
                        padding='same'
                    )
                )
        self.dropout = nn.Dropout3d(dropout)
        self.linear = nn.Linear(depths[-1], 1)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
                nn.MaxPool3d(2, stride=maxpool_stride),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    
    def forward(self, x):
        x_f = self.feature_extractor(x)
        x = F.avg_pool3d(x_f, (x_f.size(2), x_f.size(3), x_f.size(4)))
        x = self.dropout(x)
        x = x.reshape(x.size(0), x.size(1))
        x = self.linear(x)
        if self.include_top:
            return x
        if self.prediction_range is not None:
            x = torch.clamp(x, self.prediction_range[0], self.prediction_range[1])
        return x