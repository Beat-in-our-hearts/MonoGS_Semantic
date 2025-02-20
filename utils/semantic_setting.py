import yaml
from dataclasses import asdict, dataclass, field
from typing import List, Dict

@dataclass
class Semantic_Config_DataClass:
    wandb_project:str = "GSFF_SLAM"
    save_root_dir:str = "results/replica"
    mode_list: List[str] = field(default_factory=lambda: ["SAM2", "CLIP", "GT_Label", "SAM_CLIP", "Grounding_Dino", "Base_Model_Pipe"])
    mode:str = "Base_Model_Pipe" # "Base_Model_Pipe"
    enable:bool = True
    use_lseg:bool = False
    wandb_enable:bool = False
    
    eval_segmentation:bool = True
    
    gs_init_lr:float = 5.0
    semantic_lr_scale:float = 2.0
    
    semantic_window_select = {"Grounding_Dino":1, "GT_Label": 2, "Base_Model_Pipe":5}
    semantic_window:int = semantic_window_select[mode]
    
    init_iter_dict = {"Grounding_Dino": 20, "GT_Label": 10, "Base_Model_Pipe": 20}
    map_iter_dict = {"Grounding_Dino": 6, "GT_Label": 3, "Base_Model_Pipe": 1}
    
    semantic_init_iter:int = init_iter_dict[mode]
    semantic_iter:int = map_iter_dict[mode]
    semantic_threshold:float = 0.6
    
    semantic_dim_dict: Dict[str, int] = field(default_factory=lambda: {
        "LSeg": 512,
        "SAM2": 256,
        "CLIP": 768, 
        "GT_Label": 128,
        "SAM_CLIP": 512,
        "Grounding_Dino": 512,
        "Base_Model_Pipe": 512,
    })
    semantic_dim = semantic_dim_dict[mode]
    
    fmap_size: Dict[str, List[int]] = field(default_factory=lambda: {
        "LSeg": [360, 480],
        "SAM2": [64, 64], # 256, 64, 64
        "CLIP": [24, 42], # 768, 24, 42, 
    })
    dataset_path_dict: Dict[str, str] = field(default_factory=lambda: {
        "LSeg": "rgb_feature_lseg",
        "SAM2": "rgb_feature_sam2",
        "CLIP": "rgb_feature_clip",
        "GT_Label": "gt_label",
        "SAM_CLIP": "florence2_sam2", # fusion_features_0000.pt, mask_auto_label_0000.png
        "Grounding_Dino": "grounding_dino", # grounding_dino_feature_0000.pt, pred_label_0000.png
        "Base_Model_Pipe": "yolo_sam_2_18", # .pt .png # grounding_dino_v2_2_16, yolo_sam_2_18, yolo_sam_2_19
    })
    dataset_path = dataset_path_dict[mode]
    render_size = [360, 480]
    
    Debug:bool = False
    log_file:str = "results/slam.log"
    
    delete_save_dir:bool = False
    Pose_BA_flag:bool = True
    kf_only_iou:bool = True
    preload_semantic:bool = False
    gui_torch_mp:bool = False
    synchronize:bool = False
    constant_velocity_warmup:int = 5
    
    Semantic_Debug = {
        "random_select": True,
    }
    
    GT_Exp = {
        "mode": "None", # ["Sparse GT", "Noise GT"]
        "sparse_ratio": 0.01,
        "noise_ratio": 0.99,
    }
    
    Autoencoder_Test = False
    using_top_dim = True
    train_decoder = True

Semantic_Config = Semantic_Config_DataClass()

def config_to_dict(config: Semantic_Config_DataClass):
    """
    Convert a Semantic_Config dataclass to a dictionary.
    Args:
        config: An instance of the Semantic_Config dataclass.
    Returns:
        A dictionary representation of the Semantic_Config dataclass.
    """
    return asdict(config)


    