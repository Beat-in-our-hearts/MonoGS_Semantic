import os
import wandb
import yaml
from utils.semantic_setting import Semantic_Config

def wandb_init(config, save_dir, wandb_run_name, resume_flag):
    wandb_run_id = None
    wandb_config_path = os.path.join(save_dir, "wandb.yml")
    if resume_flag and os.path.exists(wandb_config_path):
        with open(wandb_config_path, "r") as yml:
            wandb_config = yaml.safe_load(yml)
        wandb_run_id = wandb_config["run_id"]
    run = wandb.init(
            project=Semantic_Config.wandb_project,
            name=wandb_run_name,
            config=config,
            id=wandb_run_id,
            mode=None if config["Results"]["use_wandb"] else "disabled",
    )
    with open(wandb_config_path, "w") as file:
        documents = yaml.dump({"run_id": run.id}, file)
        
    wandb.define_metric("frame_idx")
    wandb.define_metric("ate*", step_metric="frame_idx")
