import torch
from collections import OrderedDict
# net = torch.load('D:/code/mmpretrain-main/OrderedDict.pth')
# model = torch.load('D:/code/mmpretrain-main/net.pth')
# print(net)
#
#
#
# # net = torch.load('D:/code/mmpretrain-main/new_state_dict.pth')
# # # 假设state_dict_dict是你的普通字典
# # state_dict_ordered = OrderedDict(net)
# # torch.save(state_dict_ordered, 'OrderedDict.pth')

import torch
from mmpretrain import get_model

model = get_model('vit-base-p16_sam-pre_3rdparty_sa1b-1024px', pretrained=True,  backbone=dict(frozen_stages=12))
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))

