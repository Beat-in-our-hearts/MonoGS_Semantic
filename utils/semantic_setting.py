from dataclasses import dataclass

@dataclass
class Semantic_Config:
    mode = "SAM2"
    enable = False
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
    semantic_window = 5
    semantic_iter = 5
    render_enable = False
    gs_init_lr = 10.0
    