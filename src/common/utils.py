import einops
import torch


def get_best_model(cfg):
    return cfg.log.log_model + cfg.log.exp_name + '.pth'


def divide_in_patches(image, patch_size, stride_patches):
    patches = image.unfold(1, patch_size, stride_patches).unfold(2, patch_size, stride_patches)
    patches = einops.rearrange(patches, 'c p1 p2 h w -> (p1 p2) c h w')
    return patches


def patch_center_position(patches, original_img_size, stride):
    patch_size = patches.shape[2]
    n = int((original_img_size - patch_size) / stride + 1)
    offset = patch_size // 2
    patch_centers = []
    for i in range(n):
        patch_center_row = offset + i * stride
        for j in range(n):
            patch_center_col = offset + j * stride
            patch_centers.append([patch_center_row, patch_center_col])
    return torch.Tensor(patch_centers)
