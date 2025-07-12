from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.dataset import Compose, default_collate

from mmpretrain.apis import inference_model, init_model
from mmpretrain.utils import register_all_modules

imagenet_mean = np.array([0.15653239, 0.15653239, 0.15653239])
imagenet_std = np.array([0.27153102, 0.27153102, 0.27153102])


def show_image(img: torch.Tensor, title: str = '') -> None:
    # image is [H, W, 3]
    assert img.shape[2] == 3

    plt.imshow(img)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def save_images(original_img: torch.Tensor, img_masked: torch.Tensor,
                pred_img: torch.Tensor, img_paste: torch.Tensor,
                out_file: Optional[str] =None) -> None:
    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 6]

    plt.subplot(1, 4, 1)
    show_image(original_img, 'original')

    plt.subplot(1, 4, 2)
    show_image(img_masked, 'masked')

    plt.subplot(1, 4, 3)
    show_image(pred_img, 'reconstruction')

    plt.subplot(1, 4, 4)
    show_image(img_paste, 'reconstruction + visible')

    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
        print(f'Images are saved to {out_file}')


def recover_norm(img: torch.Tensor,
                 mean: np.ndarray = imagenet_mean,
                 std: np.ndarray = imagenet_std):
    if mean is not None and std is not None:
        img = torch.clip((img * std + mean) * 255, 0, 255).int()
    return img


def post_process(
    original_img: torch.Tensor,
    pred_img: torch.Tensor,
    mask: torch.Tensor,
    mean: np.ndarray = imagenet_mean,
    std: np.ndarray = imagenet_std
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # channel conversion
    original_img = torch.einsum('nchw->nhwc', original_img.cpu())
    # masked image
    img_masked = original_img * (1 - mask)
    # reconstructed image pasted with visible patches
    img_paste = original_img * (1 - mask) + pred_img * mask

    # muptiply std and add mean to each image
    original_img = recover_norm(original_img[0])
    img_masked = recover_norm(img_masked[0])

    pred_img = recover_norm(pred_img[0])
    img_paste = recover_norm(img_paste[0])

    return original_img, img_masked, pred_img, img_paste


ckpt_path = r"C:\Users\huang\Desktop\mmpretrain-main\epoch_120.pth"
model = init_model(
    r'C:\Users\huang\Desktop\mmpretrain-main\configs\mae\u-mae_vit-base-p16_8xb512-amp-coslr-300e_voc.py',
    ckpt_path,
    device='cpu')
print('Model loaded.')


register_all_modules()
torch.manual_seed(2)

img_path = '20sub-gl003_dir-ax_244.png'

model.cfg.test_dataloader = dict(
    dataset=dict(pipeline=[
        dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(type='Resize', scale=(224, 224), backend='pillow'),
        dict(type='PackSelfSupInputs', meta_keys=['img_path'])
    ]))

vis_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

data = dict(img_path=img_path)
data = vis_pipeline(data)
data = default_collate([data])
img, _ = model.data_preprocessor(data, False)

# for MAE reconstruction
img_embedding = model.head.patchify(img[0])
# normalize the target image
mean = img_embedding.mean(dim=-1, keepdim=True)
std = (img_embedding.var(dim=-1, keepdim=True) + 1.e-6)**.5

# get reconstruction image
features = inference_model(model, img_path)
results = model.reconstruct(features, mean=mean, std=std)


original_target = img[0]
original_img, img_masked, pred_img, img_paste = post_process(
    original_target,
    results.pred.value,
    results.mask.value,
    mean=mean,
    std=std)

save_images(original_img, img_masked, pred_img, img_paste)