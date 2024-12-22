import yaml
from dataclasses import asdict, dataclass, field
from typing import List, Dict

@dataclass
class Semantic_Config_DataClass:
    wandb_project:str = "GSDFF_SLAM"
    save_root_dir:str = "results/replica"
    mode_list: List[str] = field(default_factory=lambda: ["SAM2", "GT_Label"])
    mode:str = "SAM2"
    enable:bool = True
    wandb_enable:bool = False
    
    gs_init_lr:float = 10.0
    
    semantic_window:int = 4
    semantic_init_iter:int = 5
    semantic_iter:int = 5
    
    semantic_dim: Dict[str, int] = field(default_factory=lambda: {
        "LSeg": 512,
        "SAM2": 256,
        "GT_Label": 128
    })
    fmap_size: Dict[str, List[int]] = field(default_factory=lambda: {
        "LSeg": [360, 480],
        "SAM2": [64, 64]
    })
    dataset_path: Dict[str, str] = field(default_factory=lambda: {
        "LSeg": "rgb_feature_lseg",
        "SAM2": "rgb_feature_sam2",
        "GT_Label": "gt_label"
    })
    render_size = [360, 480]
    
    Debug:bool = False
    log_file:str = "results/slam_sp.log"
    
    delete_save_dir:bool = False
    Pose_BA_flag:bool = True
    kf_only_iou:bool = True
    preload_semantic:bool = False
    gui_torch_mp:bool = False
    synchronize:bool = False
    constant_velocity_warmup:int = 5

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


    