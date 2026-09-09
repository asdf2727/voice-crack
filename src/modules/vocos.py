from torch import nn

from modules.convnext import ConvNeXt1D

class Vocos(nn.Sequential):
    def __init__(self,
                 channels: int,
                 layers: int,
                 kernel: int = 7,
                 hidden: int | None = None):
        scale = 1 / layers
        super().__init__(*[ConvNeXt1D(channels, kernel, hidden, scale) for _ in range(layers)])
        self.latency = (kernel - 1) * layers
