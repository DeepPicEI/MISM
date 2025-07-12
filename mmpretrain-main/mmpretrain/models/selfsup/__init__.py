# Copyright (c) OpenMMLab. All rights reserved.
from .barlowtwins import BarlowTwins
from .base import BaseSelfSupervisor
from .beit import VQKD, BEiT, BEiTPretrainViT
from .byol import BYOL
from .cae import CAE, CAEPretrainViT, DALLEEncoder
from .densecl import DenseCL
from .eva import EVA
from .itpn import iTPN, iTPNHiViT
from .mae import MAE, MAEHiViT, MAEViT
from .maskfeat import HOGGenerator, MaskFeat, MaskFeatViT
from .mff import MFF, MFFViT
from .milan import MILAN, CLIPGenerator, MILANViT
from .mixmim import MixMIM, MixMIMPretrainTransformer
from .moco import MoCo
from .mocov3 import MoCoV3, MoCoV3ViT
from .simclr import SimCLR
from .simmim import SimMIM, SimMIMSwinTransformer
from .simsiam import SimSiam
from .spark import SparK
from .swav import SwAV
from .mae_u import UMAE, UMAEVIT
from .mae_e import UMAEencorder, UMAEBN
from .mae_fpn import UMAEFPN, UMAVITFPN
from .mae_e1 import UMAEencorder1, UMAEBN1
from .mae_e2 import RMT, RMTMAE
from .mae_swim_conv import UMAESwinconv, UMAESimConvMIM
from .mae_swim import UMAESwin, UMAESimMIM
from .agent_swin import AgentSwinTransformer, PreAgent
from .clipsam import ClipSam
from .RMT_Swin import RMT_Swin, PreRMT
__all__ = [
    'BaseSelfSupervisor',
    'BEiTPretrainViT',
    'VQKD',
    'CAEPretrainViT',
    'DALLEEncoder',
    'MAEViT',
    'MAEHiViT',
    'iTPNHiViT',
    'iTPN',
    'HOGGenerator',
    'MaskFeatViT',
    'CLIPGenerator',
    'MILANViT',
    'MixMIMPretrainTransformer',
    'MoCoV3ViT',
    'SimMIMSwinTransformer',
    'MoCo',
    'MoCoV3',
    'BYOL',
    'SimCLR',
    'SimSiam',
    'BEiT',
    'CAE',
    'MAE',
    'MaskFeat',
    'MILAN',
    'MixMIM',
    'SimMIM',
    'EVA',
    'DenseCL',
    'BarlowTwins',
    'SwAV',
    'SparK',
    'MFF',
    'MFFViT',
    'UMAE',
    'UMAEVIT',
    'UMAEencorder',
    'UMAEBN',
    'UMAEFPN',
    'UMAVITFPN',
    'UMAEencorder1',
    'UMAEBN1',
    'RMT',
    'RMTMAE',
    'UMAESimMIM',
    'UMAESwin',
    'UMAESwinconv',
    'UMAESimMIM',
    'AgentSwinTransformer',
    'PreAgent',
    'ClipSam',
    'RMT_Swin',
    'PreRMT'
]
