from dataclasses import dataclass

@dataclass
class Semantic_Config:
    wandb_project:str = "GSDFF_SLAM"
    save_root_dir:str = "results/replica"
    
    mode:str = "SAM2"
    enable:bool = False
    wandb_enable:bool = False
    save_render_enable:bool = False
    
    gs_init_lr:float = 6.0
    
    semantic_window:int = 5
    semantic_init_iter:int = 5
    semantic_iter:int = 5
    
    semantic_dim = { 
        "LSeg": 512,
        "SAM2": 256   
    }
    famp_size = {
        "LSeg": [360, 480],
        "SAM2": [64, 64]  
    }
    dataset_path = {
        "LSeg": "rgb_feature_lseg",
        "SAM2": "rgb_feature_sam2"
    }
    render_size = [360, 480]




    
    
    
    
    