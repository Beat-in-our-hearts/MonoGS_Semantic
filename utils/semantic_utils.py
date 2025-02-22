import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse import save_npz, load_npz, csr_matrix, vstack

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.semantic_setting import Semantic_Config
from diff_gaussian_rasterization import get_semantic_channels
import seaborn as sns
import scipy.ndimage as ndi

adepallete = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,255,214,0,25,194,194,102,255,0,92,0,255]
adepallete = np.array(adepallete).reshape(-1, 3)

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


def overlay_heatmaps_seaborn(heatmaps, alphas, cmaps, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    # 遍历热力图，依次绘制
    for heatmap, alpha, cmap in zip(heatmaps, alphas, cmaps):
        sns.heatmap(
            heatmap,
            cmap=cmap,
            alpha=alpha,  # 设置透明度
            cbar=False,   # 关闭单独颜色条
            square=True,  # 确保单元格为正方形
            xticklabels=False,  # 隐藏坐标
            yticklabels=False,   # 隐藏坐标
            ax=ax
        )
    plt.axis("off")
    plt.savefig("/tmp/temp_vis.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return cv2.cvtColor(cv2.imread("/tmp/temp_vis.png"), cv2.COLOR_BGR2RGB)

def apply_sim_colormap(img_embeds:torch.Tensor, text_embeds:torch.Tensor, gamma=5):
    sims = img_embeds @ text_embeds.T # H W D * D N -> H W N 
    sims = sims.squeeze()
    print(sims.shape)
    heatmaps = []
    for i in range(sims.shape[-1]):
        heatmaps.append(sims[:,:,i].detach().cpu().numpy()** gamma)
    alphas = [0.6] * sims.shape[-1]
    all_colormaps = plt.colormaps()
    cmaps = [all_colormaps[i] for i in range(0, len(all_colormaps), len(all_colormaps) // sims.shape[-1])] 
    img = overlay_heatmaps_seaborn(heatmaps, alphas, cmaps)
    return img


# ["rug", "table", "chair", "window"]
def generate_colored_mask(heatmaps, thresholded=95):
    num_heatmaps = len(heatmaps)
    mixed_mask = np.zeros((heatmaps[0].shape[0], heatmaps[0].shape[1], 3), dtype=np.float32)
    for idx, heatmap in enumerate(heatmaps):
        percentile = np.percentile(heatmap, thresholded)
        mask = heatmap > percentile
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_mask[mask == 1] = adepallete[150 // (num_heatmaps+1) * (idx+1) - 1]
        mixed_mask += colored_mask * (1/num_heatmaps)
    mixed_mask = mixed_mask.astype(np.uint8)
    return mixed_mask
    
# gamma = 5
# heatmap1 = sims[0, :,:,0].detach().cpu().numpy()** gamma 
# heatmap2 = sims[0, :,:,1].detach().cpu().numpy()** gamma
# heatmap3 = sims[0, :,:,2].detach().cpu().numpy()** gamma
# heatmap4 = sims[0, :,:,3].detach().cpu().numpy()** gamma
# heatmaps = [heatmap1, heatmap2, heatmap3, heatmap4]
# generate_colored_mask(heatmaps)

class Autoencoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, hidden_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, input_dim),
        )
    
    def forward(self, x):
        x = self.encoder(x)  
        x = self.decoder(x)  
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    
def build_decoder(mode='train', lr=0.001, model_type='cnn'):
    if model_type == 'cnn':
        pred_feature_dim = Semantic_Config.semantic_dim
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
    elif model_type == 'autoencoder':
        semantic_feature_dim = get_semantic_channels()
        autoencoder = Autoencoder(input_dim=Semantic_Config.semantic_dim, hidden_dim=semantic_feature_dim).to("cuda")
        autoencoder.eval()
        return autoencoder, None

def label_loss(pred:torch.Tensor, label:torch.Tensor) -> torch.Tensor:
    """
    Args:
        pred: (B, C, H, W)
        label: (B, H, W)
    """
    assert pred.dim() == label.dim() + 1, f"pred dim: {pred.dim()}, label dim: {label.dim()}"
    return nn.CrossEntropyLoss()(pred, label)

def save_seg_map(seg_map, filepath):
    sparse_map_list = []
    for i in range(seg_map.shape[0]):
        sparse_map_list.append(csr_matrix(seg_map[i]))
    save_npz(filepath, vstack(sparse_map_list))
    
def load_seg_map(filepath, W, H):
    sparse_matrix = load_npz(filepath)

    total_rows = sparse_matrix.shape[0]
    if total_rows % W != 0:
        raise ValueError("Stored sparse matrix shape is inconsistent with the provided W and H.")
    N = total_rows // W
    
    dense_seg_map = np.stack([
        sparse_matrix[i * W:(i + 1) * W].toarray()
        for i in range(N)
    ], axis=0)

    return dense_seg_map

@torch.no_grad()
def create_dense_feature(label_map, feature, dim=512) -> torch.Tensor:
    """
        return dense feature map with shape (D, H, W)
    """
    clip_features = feature["text_feature"]
    # clip_features = feature["image_feature"]
    clip_features /= clip_features.norm(dim=-1, keepdim=True)
    clip_features = clip_features.to(torch.float32).cuda() # N D
    
    dense_feature = torch.zeros((label_map.shape[0], label_map.shape[1], dim), dtype=torch.float32).cuda()
    for i in range(clip_features.shape[0]):
        dense_feature[label_map == i+1] = clip_features[i]
    return dense_feature.permute(2, 0, 1)

def cosine_similarity_map(A, B, epsilon=1e-6):
    norm_A = torch.norm(A, dim=-1, keepdim=True)  # (W, H, 1)
    norm_B = torch.norm(B, dim=-1, keepdim=True)  # (N, 1)
    
    norm_A = torch.where(norm_A < epsilon, torch.ones_like(norm_A), norm_A)
    norm_B = torch.where(norm_B < epsilon, torch.ones_like(norm_B), norm_B)
    
    dot_product = torch.einsum('whd,nd->whn', A, B) 
    similarity_map = dot_product / (norm_A * norm_B.T)  # (W, H, N)
    return similarity_map

@torch.no_grad()
def fix_pred_feature(pred_feature, empty_mask, threshold=1000):
    """
        input: 
            pred_feature: N x H x W, feature map of image
            empty_mask: H x W, boolean mask indicating empty areas (1: empty, 0: not empty)
        return: 
            Fixed pred_feature with holes filled 
    """
    
    # find small hole
    labeled_array, num_area = ndi.label(empty_mask)
    area_sizes = np.bincount(labeled_array.ravel()) 
    small_area_indices = np.where(area_sizes < threshold)[0]
    hole_mask = np.zeros_like(empty_mask)
    for indices in small_area_indices:
        hole_mask[labeled_array==indices] = 1
    hole_mask = hole_mask.astype(bool)

    fixed_feature = pred_feature
    fixed_empty_mask = empty_mask & ~hole_mask
    
    # # find hole
    # fixed_feature = pred_feature.clone()
    # hole_mask = empty_mask & valid_mask
    # fixed_feature[hole_mask] = 0
    
    # # hole repaired
    # kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
    # padded_feature = F.pad(fixed_feature, (kernel_size//2,)*4, mode='reflect')
    # neighbor_sum = F.conv2d(padded_feature.unsqueeze(1), kernel).squeeze(1)
    
    # padded_mask = F.pad((~hole_mask).float(), (kernel_size//2,)*4, mode='constant', value=0)
    # neighbor_count = F.conv2d(padded_mask.unsqueeze(0).unsqueeze(0), kernel,padding=0).squeeze()
    # avg_feature = neighbor_sum / (neighbor_count + 1e-6)
    # fixed_feature[hole_mask] = avg_feature[hole_mask]
    
    return fixed_feature, fixed_empty_mask