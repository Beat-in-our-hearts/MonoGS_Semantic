import os
import numpy as np
import torch
import torch.nn.functional as F
import encoding.utils as utils
from encoding.models.sseg import BaseNet
from modules.lseg_module import LSegModule
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from additional_utils.encoding_models import MultiEvalModule as LSeg_MultiEvalModule
from collections import namedtuple

# LSeg_args = namedtuple('LSeg_args', ['model', 'backbone', 'dataset', 'workers', 'base_size',
#                                      'crop_size', 'train_split', 'aux', 'se_loss', 'se_weight',
#                                      'batch_size', 'test_batch_size', 'no_cuda', 'seed', 'weights',
#                                      'eval', 'export', 'acc_bn', 'test_val', 'no_val', 'module',
#                                      'data_path', 'no_scaleinv', 'widehead', 'widehead_hr', 'ignore_index',
#                                      'label_src', 'jobname', 'no_strict', 'arch_option', 'block_depth',
#                                      'activation', 'outdir', 'test_rgb_dir', 'resize_max'])


LSeg_args = namedtuple('LSeg_args', ['weights', 'data_path', 'dataset', 'backbone', 
                                     'aux', 'ignore_index', 'scale_inv', 'widehead',
                                     'widehead_hr', 'img_size'])

adepallete = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,255,214,0,25,194,194,102,255,0,92,0,255]



def load(checkpoint_path, config, device):
    pass
    
def get_legend_patch(npimg, new_palette, labels):
    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    out_img.putpalette(new_palette)
    u_index = np.unique(npimg)
    patches = []
    for i, index in enumerate(u_index):
        label = labels[index]
        cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0, new_palette[index * 3 + 2] / 255.0]
        red_patch = mpatches.Patch(color=cur_color, label=label)
        patches.append(red_patch)
    return out_img, patches

class LSeg_FeatureExtractor(torch.nn.Module):
    def __init__(self, debug=False):
        super(LSeg_FeatureExtractor, self).__init__()
        args = LSeg_args(weights='demo_e200.ckpt', 
                        data_path=None, 
                        dataset='ignore', 
                        backbone='clip_vitl16_384',
                        aux=False,
                        ignore_index = 255,
                        scale_inv=False,
                        widehead=True,
                        widehead_hr=False,
                        img_size=[480, 360])
        
        module = LSegModule.load_from_checkpoint(
            checkpoint_path=args.weights,
            data_path=args.data_path,
            dataset=args.dataset,
            backbone=args.backbone,
            aux=args.aux,
            num_features=256,
            aux_weight=0,
            se_loss=False,
            se_weight=0,
            base_lr=0,
            batch_size=1,
            max_epochs=0,
            ignore_index=args.ignore_index,
            dropout=0.0,
            scale_inv=args.scale_inv,
            augment=False,
            no_batchnorm=False,
            widehead=args.widehead,
            widehead_hr=args.widehead_hr,
            map_locatin="cpu",
            arch_option=0,
            block_depth=0,
            activation='lrelu',
        )
        self.labels = module.get_labels('ade20k')
        self.input_transform = module.val_transform
        self.num_classes = len(self.labels)
        
        if isinstance(module.net, BaseNet):
            model = module.net
        else:
            model = module
            
        model = model.eval()
        model = model.cpu()
        print(model)
        
        self.scales = [0.75, 1.0, 1.25, 1.75]
        self.img_size = args.img_size
        print("scales: ", self.scales)
        print("img_size: ", self.img_size)
        
        self.evaluator = LSeg_MultiEvalModule(model, self.num_classes, scales=self.scales, flip=True).cuda()
        self.evaluator.eval()
        
        self.debug = debug
    
    def _log(self, text):
        if self.debug:
            print(text)
    
    @torch.no_grad()
    def preprocess(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            image = self.input_transform(image).unsqueeze(0)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image = self.input_transform(image).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            pass 
        else:
            raise ValueError("Unsupported input type. Supported types: str (file path), numpy.ndarray, torch.Tensor")
        self._log(f"input size: {image.shape}\n")
        image_tensor = F.interpolate(image, size=(self.img_size[1], self.img_size[0]),
                                      mode="bilinear", align_corners=True)
        self._log(f"resize size: {image_tensor.shape}\n")
        return image_tensor
    
    @torch.no_grad()
    def forward_feature(self, image: torch.Tensor):
        output_features = self.evaluator.parallel_forward(image, return_feature=True)
        return output_features[0].cpu().numpy().astype(np.float16)
    
    @torch.no_grad()
    def forward(self, image):
        image_tensor = self.preprocess(image)
        return self.forward_feature(image_tensor)

    @torch.no_grad()
    def vis_feature(self, image, outname='test', outdir='vis'):
        image_tensor = self.preprocess(image)
        outputs = self.evaluator.parallel_forward(image_tensor)[0]
        predicts = torch.max(outputs, 1)[1].cpu().numpy()
        
        # save mask
        masks = utils.get_mask_pallete(predicts, 'detail')
        masks.save(os.path.join(outdir, outname+'.png'))
        
        # save vis
        masks_tensor = torch.tensor(np.array(masks.convert("RGB"), "f")) / 255.0
        vis_img = (image_tensor[0] + 1) / 2.
        vis_img = vis_img.permute(1, 2, 0)  # ->hwc
        vis1 = vis_img
        vis2 = vis_img * 0.4 + masks_tensor * 0.6
        vis3 = masks_tensor
        vis = torch.cat([vis1, vis2, vis3], dim=1)
        Image.fromarray((vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(outdir, outname+"_vis.png"))

        # save label vis
        seg, patches = get_legend_patch(predicts, adepallete, self.labels)
        seg = seg.convert("RGBA")
        plt.figure()
        plt.axis('off')
        plt.imshow(seg)
        plt.legend(handles=patches, prop={'size': 8}, ncol=4)
        plt.savefig(os.path.join(outdir, outname+"_legend.png"), format="png", dpi=300, bbox_inches="tight")
        plt.clf()
        plt.close()