import torch
from mmpretrain import get_model
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class SAM(BaseModule):
    def __init__(self):
        super(SAM, self).__init__()
        # self.sam = get_model('vit-base-p16_sam-pre_3rdparty_sa1b-1024px', pretrained=True,  backbone=dict(frozen_stages=12))
        self.sam = get_model('vit-base-p16_sam-pre_3rdparty_sa1b-1024px', pretrained=True)

    def forward(self, img):
        out = self.sam(img)
        return out