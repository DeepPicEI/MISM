import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from mmpretrain import get_model


# @MODELS.register_module()
class CLIPSamHead(BaseModule):
    """Head for SimMIM Pre-training.

    Args:
        patch_size (int): Patch size of each token.
        loss (dict): The config for loss.
    """

    def __init__(self, loss=dict(type='PixelReconstructionLoss', criterion='L2', channel=3)) -> None:
        super().__init__()
        self.sam = get_model('vit-base-p16_sam-pre_3rdparty_sa1b-1024px', pretrained=True, backbone=dict(frozen_stages=12))

        self.loss_module = MODELS.build(loss)

    def loss(self, pred: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Generate loss.

        This method will expand mask to the size of the original image.

        Args:
            pred (torch.Tensor): The reconstructed image (B, C, H, W).
            target (torch.Tensor): The target image (B, C, H, W).
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        traget = self.sam(image)
        loss = self.loss_module(pred, traget[0])

        return loss

if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    pred = torch.randn(2, 256, 14, 14)
    net = CLIPSamHead()
    out = net.loss(pred, x)
    print(out)