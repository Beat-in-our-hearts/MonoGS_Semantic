from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

import argparse
import os

from collections import namedtuple


SAM_args = namedtuple('SAM_args', ['model_type', 'checkpoint'])

def sam_mask_visualize(masks):
    if len(masks) == 0:
        raise ValueError("no mask")
    shape = masks[0]['segmentation'].shape
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    colors = np.random.randint(0, 256, size=(len(masks), 3))
    for idx, mask in enumerate(masks):
        color = colors[idx]
        seg = mask['segmentation']
        canvas[seg] = color
    return canvas

def sam_quick_show_image(rgb_image, masks):
    """slow draw"""
    if len(masks) == 0:
        raise ValueError("no mask")
    colors = np.random.rand(len(masks), 3)
    shape = masks[0]['segmentation'].shape
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    colors = np.random.randint(0, 256, size=(len(masks), 3))
    for idx, mask in enumerate(masks):
        color = colors[idx]
        seg = mask['segmentation']
        canvas[seg] = color
    mix_image = (0.7 * rgb_image + 0.3 * canvas).astype(np.uint8)
    plt.figure(figsize=(10,10))
    plt.imshow(mix_image)       
    plt.axis('off')
    plt.title("Qucik Overlay Color Visualization")
    plt.show()
    
    
class SAM_FeatureExtractor:
    def __init__(self, debug=False):
        args = SAM_args(model_type='vit_h', 
                        checkpoint='checkpoints/sam_vit_h_4b8939.pth')
        self.sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).cuda()
        self.predictor = SamPredictor(self.sam) 
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        
    @torch.no_grad()
    def preprocess(self, image) -> np.ndarray:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            pass
        elif isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        else:
            raise ValueError('Image input type not recognized')
        assert image.shape[-1] == 3 and len(image.shape) == 3, 'Input image must have 3 channels, shape (H, W, 3)'
        return image
    
    @torch.no_grad()
    def forward_feature(self, image:np.ndarray) -> np.ndarray:
        self.predictor.set_image(image)
        image_embedding_numpy = self.predictor.get_image_embedding().squeeze(0).cpu().numpy()
        img_h, img_w, _ = image.shape
        _, feat_h, feat_w = image_embedding_numpy.shape
        if img_h <= img_w:
            cropped_h = int(feat_w / img_w * img_h + 0.5)
            image_embedding_numpy_cropped = image_embedding_numpy[:, :cropped_h, :]
        else:
            cropped_w = int(feat_h / img_h * img_w + 0.5)
            image_embedding_numpy_cropped = image_embedding_numpy[:, :, :cropped_w]
        return image_embedding_numpy_cropped, image_embedding_numpy
        
    @torch.no_grad()
    def features_to_image(self, image_embedding_tensor: torch.Tensor, fake_image: np.ndarray):
        masks = self.mask_generator.generate(fake_image, features=image_embedding_tensor)
        rgb_render = sam_mask_visualize(masks)
        output = {"rgb_render": rgb_render}
        return output
        
    @torch.no_grad()
    def extract_feature(self, image):
        image = self.preprocess(image)
        image_embedding_numpy_cropped, image_embedding_numpy = self.forward_feature(image)
        image_embedding_tensor = torch.tensor(image_embedding_numpy).cuda()
        vis_out = self.features_to_image(image_embedding_tensor, image)
        output = {"numpy_feature": image_embedding_numpy_cropped, "feat_type": "sam",
                      "rgb_render": vis_out["rgb_render"]}
        return output