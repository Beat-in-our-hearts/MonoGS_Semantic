import torch
import torch.nn as nn

from utils.semantic_setting import Semantic_Config
from diff_gaussian_rasterization import get_semantic_channels

def apply_pca_colormap_return_proj(
    image:torch.Tensor,
    proj_V = None,
    low_rank_min = None,
    low_rank_max = None,
    niter: int = 5,
):
    """Convert a multichannel image to color using PCA.

    Args:
        image: Multichannel image.
        proj_V: Projection matrix to use. If None, use torch low rank PCA.

    Returns:
        Colored PCA image of the multichannel input image.
    """
    image_flat = image.reshape(-1, image.shape[-1])

    # Modified from https://github.com/pfnet-research/distilled-feature-fields/blob/master/train.py
    if proj_V is None:
        mean = image_flat.mean(0)
        with torch.no_grad():
            U, S, V = torch.pca_lowrank(image_flat - mean, niter=niter)
        proj_V = V[:, :3]

    low_rank = image_flat @ proj_V
    if low_rank_min is None:
        low_rank_min = torch.quantile(low_rank, 0.01, dim=0)
    if low_rank_max is None:
        low_rank_max = torch.quantile(low_rank, 0.99, dim=0)

    low_rank = (low_rank - low_rank_min) / (low_rank_max - low_rank_min)
    low_rank = torch.clamp(low_rank, 0, 1)

    colored_image = low_rank.reshape(image.shape[:-1] + (3,))
    return colored_image, proj_V, low_rank_min, low_rank_max

def apply_pca_colormap(
    image:torch.Tensor,
    proj_V = None,
    low_rank_min = None,
    low_rank_max = None,
    niter: int = 5,
):
    return apply_pca_colormap_return_proj(image, proj_V, low_rank_min, low_rank_max, niter)[0]


def build_decoder(mode='train', lr=0.0005):
    pred_feature_dim = Semantic_Config.semantic_dim[Semantic_Config.mode]
    semantic_feature_dim = get_semantic_channels()
    cnn_decoder, cnn_decoder_optimizer = None, None
    if mode == 'train':
        cnn_decoder = nn.Conv2d(semantic_feature_dim, pred_feature_dim, kernel_size=1).to("cuda")
        cnn_decoder.requires_grad_(True)
        cnn_decoder_optimizer = torch.optim.Adam(cnn_decoder.parameters(), lr=lr)
    elif mode == 'eval':
        cnn_decoder = nn.Conv2d(semantic_feature_dim, pred_feature_dim, kernel_size=1).to("cuda")
        cnn_decoder.eval()
    return cnn_decoder, cnn_decoder_optimizer