import os
import wandb
from typing import Any
import numpy as np
import torch


class WandbWriter:
    def __init__(self, project=None, run_name=None, resume=False, run_id =None):
        r"""A Wrapper for wandb.

        Args:
            *args: Additional positional args for SummaryWriter
            **kwargs: Additional keyword args for SummaryWriter
        """
        
        self.use_wandb = False
        if project != None:
            self.use_wandb = True
            
        if self.use_wandb: 
            os.system("wandb login --relogin $WANDB_API_KEY")
            if resume:
                self.run = wandb.init(project=project, id=run_id, resume="must")
            else:
                self.run = wandb.init(project=project, name=run_name)
            
    def __enter__(self):
        return self

    def __exit__(self):
        if self.use_wandb:
            self.run.finish()
            
    def add_video_from_np_images(self, video_name: str, step_idx: int, images: np.ndarray, fps: int = 10) -> None:
        r"""Write video into wandb from images frames.

        Args:
            video_name: name of video string.
            step_idx: int of checkpoint index to be displayed.
            images: list of n frames. Each frame is a np.ndarray of shape.
            fps: frame per second for output video.

        Returns:
            None.
        """
        if self.use_wandb:
            # initial shape of np.ndarray list: N * (H, W, 3)
            frame_tensors = [torch.from_numpy(np_arr).unsqueeze(0) for np_arr in images]
            video_tensor = torch.cat(tuple(frame_tensors))
            video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
            # final shape of video tensor: (1, n, 3, H, W)

            video = wandb.Video(data_or_path=video_tensor, caption=video_name, fps=fps)
            wandb.log({video_name: video}, step=int(step_idx))

    def add_image(self, descriptor: str, img: Any, global_step: int):
        if len(img.shape) < 3:
            img = img.unsqueeze(-1)
        if self.use_wandb:
            img = wandb.Image(img.permute(2,0,1), caption="")
            wandb.log({descriptor: img}, step=int(global_step))

    def add_pil_image(self, descriptor: str, img: Any, global_step: int):
        if self.use_wandb:
            img = wandb.Image(img, caption="")
            wandb.log({descriptor: img}, step=int(global_step))

    def add_scalar(self, descriptor: str, value: Any, global_step: int):
        if self.use_wandb:
            wandb.log({descriptor: value}, step=int(global_step))

    def add_scalars(self, descriptor: str, values: dict, global_step: int):
        for k in values:
            self.add_scalar(descriptor+k, values[k], global_step)