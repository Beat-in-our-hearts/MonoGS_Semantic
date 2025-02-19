import json
import os
import shutil
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians, eval_segmentation
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd

from utils.semantic_setting import Semantic_Config, config_to_dict
from utils.wandb_utils import wandb_init
import clip

class SLAM:
    def __init__(self, config, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.track_gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        
        self.gaussians.init_lr(Semantic_Config.gs_init_lr, Semantic_Config.semantic_lr_scale)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.gaussians = self.track_gaussians
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.use_gui = self.use_gui
        self.frontend.set_hyperparams()
        # NOTE : Load CLIP model
        if Semantic_Config.mode != "GT_Label":
            self.frontend.clip_model, _ = clip.load("ViT-B/32", device=self.frontend.device, 
                                            jit=True, download_root="/tmp")
            self.frontend.clip_model.eval()
            
            if self.config["Dataset"]["type"] in ["replica", "replica_semantic"]:
                with open("gui/info_semantic.json", "r") as f:
                    info_semantic = json.load(f) 
                class_names = [item["name"] for item in info_semantic["classes"]]
            
            elif self.config["Dataset"]["type"] in ["tum", "tum_semantic"]:
                class_names = ['chair', 'toy', 'mouse', 'telephone', 'indoor plant', 'tool', 'keyboard', 'monitor', 'tape', 'bin', 'bottle', 'ball', 'cup', 'picture', 'paper', 'book', 'notebook computer']
            
            gt_text_tokens = clip.tokenize(class_names).to(self.frontend.device)
            gt_text_features = self.frontend.clip_model.encode_text(gt_text_tokens)
            gt_text_features /= gt_text_features.norm(dim=-1, keepdim=True)
            self.frontend.gt_text_features = gt_text_features.to(torch.float32).detach()
            self.backend.gt_text_features = gt_text_features.to(torch.float32).detach()
        
        self.backend.dataset = self.dataset
        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()
        
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            all_frame_id = list(range(self.frontend.kf_indices[-1]))
            ate_result = eval_ate(
                self.frontend.cameras,
                all_frame_id,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
                depth_l1=not self.monocular,
            )
            if Semantic_Config.eval_segmentation and Semantic_Config.enable:
                if Semantic_Config.mode == "GT_Label":
                    cnn_decoder_state_dict = None
                else:
                    cnn_decoder_state_dict = self.frontend.cnn_decoder.state_dict()
                if self.config["Dataset"]["type"] in ["replica", "replica_semantic", "scannet", "scannet_semantic"]:
                    seg_result = eval_segmentation(
                                self.frontend.cameras,
                                self.dataset,
                                self.gaussians,
                                self.pipeline_params,
                                self.background,
                                self.save_dir,
                                cnn_decoder_state_dict,
                                self.frontend.gt_text_features,
                            )
            else:
                seg_result = {"pixel_acc": 0, "mIoU": 0}
            columns = ["scene_name", "tag", "Render_Metrics", "ATE_Metrics", "Seg_Metrics", "FPS"]
            normalized_path = os.path.normpath(self.config["Dataset"]["dataset_path"])
            scene_name = os.path.basename(normalized_path)
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                scene_name,
                "Before",
                rendering_result,
                ate_result,
                seg_result,
                FPS,
            )

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    self.gaussians.load_state_dict(data[1])
                    break

            rendering_result_after = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
                depth_l1=not self.monocular,
            )
            metrics_table.add_data(
                scene_name,
                "After",
                rendering_result_after,
                ate_result,
                seg_result,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            
            final_res = {"ate_result": ate_result,
                         "rendering_result": rendering_result,
                         "seg_result": seg_result,
                         "rendering_result_after": rendering_result_after}
            
            with open(os.path.join(save_dir, "metric", "eval_final.json"), 'w', encoding='utf-8') as f:
                json.dump(final_res, f, indent=4)   

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)
    config = load_config(args.config)
    
    semantic_config_dict = config_to_dict(Semantic_Config)
    config["Semantic_Config"] = semantic_config_dict
    
    # render
    if args.render:
        config["Results"]["eval_rendering"] = True
    # headless 
    if args.headless:
        config["Results"]["save_results"] = True
        config["Results"]["use_gui"] = False
        config["Results"]["use_wandb"] = False
        Log("Running MonoGS in Headless Mode")
        Log("Following config will be overriden")
    Log(f"\tsave_results={config['Results']['save_results']}")
    Log(f"\tuse_gui={config['Results']['use_gui']}")
    Log(f"\teval_rendering={config['Results']['eval_rendering']}")
    Log(f"\tuse_wandb={config['Results']['use_wandb']}")

    # set save dir
    save_dir = None
    if config["Results"]["save_results"]:
        if args.save_path:
            save_dir = args.save_path
            config["Results"]["save_dir"] = save_dir
        elif Semantic_Config.save_root_dir is not None:
            scene_name = config["Dataset"]["dataset_path"].split("/")[-1] 
            save_dir = os.path.join(Semantic_Config.save_root_dir, scene_name)
            config["Results"]["save_dir"] = save_dir
        else: # auto set save path
            path = config["Dataset"]["dataset_path"].split("/")
            save_dir = os.path.join(config["Results"]["save_dir"], path[-2] + "_" + path[-1])
            config["Results"]["save_dir"] = save_dir
            
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        
        # set wandb
        wandb_name = args.config.split(".")[0]
        wandb_init(config, save_dir, wandb_name, args.resume)
        
        if Semantic_Config.delete_save_dir:
            Log("Deleting save_dir")
            shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)

    # run
    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    Log(f"Time: {start_time}")
    
    slam = SLAM(config, save_dir=save_dir)
    slam.run()
    wandb.finish()
    
    # relog
    end_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    Log(f"Time: {end_time}")
    Log("saving results in " + save_dir)

    # All done
    Log("Done.")
