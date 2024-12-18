import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

import random
import time
from tqdm import tqdm
from typing import Dict, List, Union

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
from utils.camera_utils import Camera

from utils.common_var import *

class BackEnd_Map(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians:GaussianModel = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue:mp.Queue = None
        self.backend_queue:mp.Queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.idle_update = 0

        self.occ_aware_visibility = {}
        self.cameras:Dict[int, Camera] = dict()
        self.current_window:List[int] = []
        self.initialized = not self.monocular
        self.keyframe_pose_optimizer:Optimizer = None
        
        self.semantic_flag = False

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
        
        self.frames_to_optimize = self.config["Training"]["pose_window"]
        self.prune_mode = self.config["Training"]["prune_mode"]
        
    def set_semantic_decoder(self):
        if self.semantic_flag:
            # CNN Decoder to upsample semantic features
            self.cnn_decoder = nn.Conv2d(SEMANTIC_FEATURES_DIM, LSeg_FEATURES_DIM, kernel_size=1).to(self.device)
            self.cnn_decoder_optimizer = torch.optim.Adam(self.cnn_decoder.parameters(), lr=0.0005)

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map)
    
    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.cameras = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_pose_optimizer = None
        
        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()
            
    def color_refinement(self):
        Log("Starting color refinement")
        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.cameras.keys())
            random_frame_idx = viewpoint_idx_stack.pop(random.randint(0, len(viewpoint_idx_stack) - 1))
            viewpoint = self.cameras[random_frame_idx]
            render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
            image, visibility_filter, radii = (render_pkg["render"], render_pkg["visibility_filter"], render_pkg["radii"])
            # Compute loss
            gt_image = viewpoint.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (Ll1) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")
                
    def push_to_frontend(self, tag=None):
        # if semantic_flag is True, save the ckpt in gaussians
        if self.semantic_flag:
            decoder_state_dict = self.cnn_decoder.state_dict()
            state_dict_cpu = {key: value.cpu() for key, value in decoder_state_dict.items()}
            self.gaussians.semantic_decoder = state_dict_cpu
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.cameras[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes] # TODO big gaussians 
        self.frontend_queue.put(msg)
        
    def update_track_optimizer(self, frames_to_optimize) -> Optimizer:
        opt_params = []
        for cam_idx in range(len(self.current_window)):
            if self.current_window[cam_idx] == 0:
                continue
            viewpoint = self.cameras[self.current_window[cam_idx]]
            if cam_idx < frames_to_optimize:
                opt_params.append(
                    {
                        "params": [viewpoint.cam_rot_delta],
                        "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                        * 0.5,
                        "name": "rot_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.cam_trans_delta],
                        "lr": self.config["Training"]["lr"][
                            "cam_trans_delta"
                        ]
                        * 0.5,
                        "name": "trans_{}".format(viewpoint.uid),
                    }
                )
            opt_params.append(
                {
                    "params": [viewpoint.exposure_a],
                    "lr": 0.01,
                    "name": "exposure_a_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.exposure_b],
                    "lr": 0.01,
                    "name": "exposure_b_{}".format(viewpoint.uid),
                }
            )
        keyframe_pose_optimizer = torch.optim.Adam(opt_params)
        return keyframe_pose_optimizer
        
    def initialize_map(self, cur_frame_idx, viewpoint:Camera):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_semantic_flag = self.semantic_flag and mapping_iteration > self.init_itr_num - MAPPING_START_ITR
            render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background, flag_semantic=render_semantic_flag)
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
                feature_map
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
                render_pkg["feature_map"]
            )
            
            loss_init = get_loss_mapping(self.config, image, depth, viewpoint, opacity, initialization=True)
            if render_semantic_flag:
                feature_map = self.cnn_decoder(F.interpolate(feature_map.unsqueeze(0), 
                                                             size=(LSeg_IMAGE_SIZE[0], LSeg_IMAGE_SIZE[1]),
                                                            mode="bilinear", align_corners=True).squeeze(0))
                if isinstance(viewpoint.semantic_feature, np.ndarray):
                    gt_feature = torch.tensor(viewpoint.semantic_feature).cuda()
                elif isinstance(viewpoint.semantic_feature, str):
                    gt_feature = torch.tensor(np.load(viewpoint.semantic_feature)).cuda()
                elif isinstance(viewpoint.semantic_feature, torch.Tensor):
                    raise Exception("Do not put torch.Tensor in Camera.semantic_feature")
                else:
                    raise Exception("Unknown semantic feature type")
                l1_feature = l1_loss(feature_map, gt_feature)
                loss_init += l1_feature
                if mapping_iteration % 50 == 0:
                    print("L1 feature loss: ", l1_feature.item())
            loss_init.backward()
            with torch.no_grad():
                # update the max radii of the gaussians
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # NOTE: gaussians densify_and_prune
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(self.opt_params.densify_grad_threshold, 
                                                     self.init_gaussian_th, self.init_gaussian_extent, None)
                # reset_opacity
                if self.iteration_count == self.init_gaussian_reset or self.iteration_count == self.opt_params.densify_from_iter:
                    self.gaussians.reset_opacity()
                
                if self.semantic_flag:
                    self.cnn_decoder_optimizer.step()
                    self.cnn_decoder_optimizer.zero_grad()
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
        
        # for each keyframe, update aware visibility    
        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()    
        Log("Initialized map")
        torch.cuda.empty_cache()
        return render_pkg
                        
    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return
        
        # get keyframes from the current window and the random viewpoints not in the window
        viewpoint_stack = [self.cameras[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = [viewpoint for cam_idx, viewpoint in self.cameras.items() if cam_idx not in current_window]
        random_update_num = 2
        
        for _ in range(iters):
            self.iteration_count += 1
            
            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []
            
            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                render_semantic_flag = self.semantic_flag and cam_idx <= 2 # TODO
                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background, flag_semantic=render_semantic_flag)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                    feature_map,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                    render_pkg["feature_map"],
                )
                loss_mapping += get_loss_mapping(self.config, image, depth, viewpoint, opacity)
                if render_semantic_flag:
                    feature_map = self.cnn_decoder(F.interpolate(feature_map.unsqueeze(0), 
                                                                size=(LSeg_IMAGE_SIZE[0], LSeg_IMAGE_SIZE[1]),
                                                                mode="bilinear", align_corners=True).squeeze(0))
                    if isinstance(viewpoint.semantic_feature, np.ndarray):
                        gt_feature = torch.tensor(viewpoint.semantic_feature).cuda()
                    elif isinstance(viewpoint.semantic_feature, str):
                        gt_feature = torch.tensor(np.load(viewpoint.semantic_feature)).cuda()
                    elif isinstance(viewpoint.semantic_feature, torch.Tensor):
                        raise Exception("Do not put torch.Tensor in Camera.semantic_feature")
                    else:
                        raise Exception("Unknown semantic feature type")
                    l1_feature = l1_loss(feature_map, gt_feature)
                    loss_mapping += l1_feature
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)
                    
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:random_update_num]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background, flag_semantic=self.semantic_flag)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                    feature_map,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                    render_pkg["feature_map"],
                )
                loss_mapping += get_loss_mapping(self.config, image, depth, viewpoint, opacity)
                if render_semantic_flag:
                    feature_map = self.cnn_decoder(F.interpolate(feature_map.unsqueeze(0), 
                                                                size=(LSeg_IMAGE_SIZE[0], LSeg_IMAGE_SIZE[1]),
                                                                mode="bilinear", align_corners=True).squeeze(0))
                    if isinstance(viewpoint.semantic_feature, np.ndarray):
                        gt_feature = torch.tensor(viewpoint.semantic_feature).cuda()
                    elif isinstance(viewpoint.semantic_feature, str):
                        gt_feature = torch.tensor(np.load(viewpoint.semantic_feature)).cuda()
                    elif isinstance(viewpoint.semantic_feature, torch.Tensor):
                        raise Exception("Do not put torch.Tensor in Camera.semantic_feature")
                    else:
                        raise Exception("Unknown semantic feature type")
                    l1_feature = l1_loss(feature_map, gt_feature)
                    loss_mapping += l1_feature
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    # for each keyframe, update aware visibility    
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()
                    
                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.window_size:
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if self.prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < prune_coviz
                            # make sure we don't split the gaussians, break here.
                        if self.prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2] if self.initialized else self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(self.gaussians.n_obs <= prune_coviz, mask)
                        
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = self.occ_aware_visibility[current_idx][~to_prune]
                                    
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # make sure we don't split the gaussians, break here.
                    return False
                
                # update the max radii of the gaussians
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(viewspace_point_tensor_acm[idx], visibility_filter_acm[idx])
                
                update_gaussian = self.iteration_count % self.gaussian_update_every == self.gaussian_update_offset
                
                ## Gaussian update densify_and_prune
                if update_gaussian:
                    self.gaussians.densify_and_prune(self.opt_params.densify_grad_threshold,
                                                     self.gaussian_th, self.gaussian_extent, self.size_threshold)
                    gaussian_split = True
                
                ## Opacity reset
                if self.iteration_count % self.gaussian_reset == 0 and not update_gaussian:
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True
                    
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_pose_optimizer.step()
                self.keyframe_pose_optimizer.zero_grad(set_to_none=True)
                if render_semantic_flag:
                    self.cnn_decoder_optimizer.step()
                    self.cnn_decoder_optimizer.zero_grad()
                    
                # Pose update
                for cam_idx in range(min(self.frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid != 0:
                        update_pose(viewpoint)
        
            torch.cuda.empty_cache()
        return gaussian_split

    def idle_map(self, current_window, prune=False, iters=1):
        self.idle_update +=1
        self.map(current_window, prune, iters)
    
    def run(self):
        self.set_semantic_decoder()
        while True:
            if self.backend_queue.empty():
                # Check if no new data has been received
                wait_pause = self.pause
                wait_empty_window = len(self.current_window) == 0
                wait_single_thread = self.single_thread
                if wait_pause or wait_empty_window or wait_single_thread:
                    time.sleep(0.01)
                    continue
                # TODO: when idle, map for few iterations
                self.idle_map(self.current_window, iters=5)
                if self.idle_update % 10 == 0:
                    start_time = time.time() 
                    self.idle_map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
                    print(f"[SLAM -- Idle Mapping] time: {time.time() - start_time}")
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    Log("Resetting the system")
                    self.reset()
                    self.cameras[cur_frame_idx] = viewpoint
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, init=True)
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")
                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    
                    self.cameras[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)
                    
                    frames_to_optimize = self.window_size
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 20
                    if not self.initialized:
                        if len(self.current_window) == self.window_size: 
                            iter_per_kf = 50 if self.live_mode else 300
                            frames_to_optimize = self.window_size -1
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    start_time = time.time()
                    self.keyframe_pose_optimizer = self.update_track_optimizer(frames_to_optimize)
                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    print(f"[SLAM -- Mapping] time: {time.time() - start_time} window: {self.current_window} iter_per_kf:{iter_per_kf}")
                    time.sleep(1)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        # empty queue
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
