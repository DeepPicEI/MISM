# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class RMTMAEHead(BaseModule):
    """Head for SimMIM Pre-training.


        loss (dict): The config for loss.
    """

    def __init__(self, loss: dict) -> None:
        super().__init__()
        self.loss_module = MODELS.build(loss)

    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        """Generate loss.

        This method will expand mask to the size of the original image.

        Args:
            pred (torch.Tensor): The reconstructed image (B, C, H, W).
            target (torch.Tensor): The target image (B, C, H, W).
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        mask = mask.unsqueeze(1).contiguous()
        loss = self.loss_module(pred, target, mask)

        return loss