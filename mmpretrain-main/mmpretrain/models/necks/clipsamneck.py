import torch
from mmengine.model import BaseModule
from mmseg.models.utils import resize
from mmpretrain.registry import MODELS
import torch.nn as nn


@MODELS.register_module()
class CLIPSamNeck(BaseModule):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, in_put: torch.Tensor) -> torch.Tensor:
        h, w = in_put.shape[2:]
        out = resize(
            input=in_put,
            size=[2*h, 2*w],
            mode='bilinear')
        out = self.conv(out)

        return out



if __name__ == '__main__':
    x = torch.randn(4, 1024, 7, 7)
    net = CLIPSamNeck(in_channels=1024, out_channels=256)
    out = net(x)
    print(out)