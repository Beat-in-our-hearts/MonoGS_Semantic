import glob
import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import yaml
import json
from munch import munchify

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.camera_utils import Camera
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth, get_loss_mapping
from utils.multiprocessing_utils import clone_obj
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from eval.eval_metrics import Eval_Tracking, Eval_Mapping, Eval_Semantic
from utils.common_var import *

class SLAM:
    def __init__(self, config, save_dir=None):
        self.config = config
        self.save_dir = save_dir
        self.__setup_params()
        self.__setup_gaussians()
        self.__setup_tracking_params()
        self.__setup_mapping_params()
        self.__setup_semantic_params()
        
    def __setup_params(self):
        self.model_params = munchify(self.config["model_params"]) 
        self.opt_params = munchify(self.config["opt_params"])
        self.pipeline_params = munchify(self.config["pipeline_params"])
        
        ###################################################
        # Set up the Dataset parameters
        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        
        ###################################################
        # Set up the Training parameters
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.config["Training"]["monocular"] = self.monocular
        
        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        # keyframe check
        self.kf_translation = self.config["Training"]["kf_translation"]
        self.kf_min_translation = self.config["Training"]["kf_min_translation"]
        self.kf_overlap = self.config["Training"]["kf_overlap"]
        # window update
        self.cut_off = (self.config["Training"]["kf_cutoff"] if "kf_cutoff" in self.config["Training"] else 0.4)
        # mapping
        self.cameras_extent = 6.0
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (self.cameras_extent * self.config["Training"]["init_gaussian_extent"])
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (self.cameras_extent * self.config["Training"]["gaussian_extent"])
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.frames_to_optimize = self.config["Training"]["pose_window"]
        self.prune_mode = self.config["Training"]["prune_mode"]
        
        ###################################################
        # Set up the Results parameters
        self.config["Results"]["save_dir"] = self.save_dir
        self.save_results = self.config["Results"]["save_results"]
        self.use_gui = True if self.live_mode else self.config["Results"]["use_gui"]
        self.eval_rendering = self.config["Results"]["eval_rendering"]
        self.model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]
        
        ###################################################
        # other local variables
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    def __setup_gaussians(self):
        # Initialize the Dataset
        self.dataset = load_dataset(self.model_params, self.model_params.source_path, config=self.config)
        
        # Initialize the Gaussian Model
        self.gaussians = GaussianModel(self.model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)
        
        # set projection_matrixv
        self.projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        self.projection_matrix = self.projection_matrix.cuda()
        
    def __setup_tracking_params(self):
        self.track_initialized = False
        
        self.cameras:Dict[int, Camera] = dict()
        self.keyframe_indices = []
        self.current_window:List[int] = []

        self.occ_aware_visibility = {}
        self.track_reset_flag = True

        self.pause = False
        
    def __setup_mapping_params(self):
        self.map_iteration_count = 0
        self.map_initialized = False
        
        
    def __setup_semantic_params(self):
        self.semantic_flag = True
        self.decoder_init = False
        self.save_semantic = True
        self.save_frame_visualize = True
        
    def semantic_init(self):
        # CNN Decoder to upsample semantic features
        self.cnn_ckpt_path = "checkpoints/cnn_decoder_dim64.pth"
        self.decoder_init = True
        self.cnn_decoder = nn.Conv2d(SEMANTIC_FEATURES_DIM, LSeg_FEATURES_DIM, kernel_size=1).to("cuda")
        self.cnn_decoder.load_state_dict(torch.load(self.cnn_ckpt_path, weights_only=True))
        self.cnn_decoder_optimizer = torch.optim.Adam(self.cnn_decoder.parameters(), lr=0.0005)
        
    def run_gui(self):
        if self.use_gui: 
            self.q_main2vis = mp.Queue() 
            self.q_vis2main = mp.Queue() 
            self.params_gui = gui_utils.ParamsGUI(
                pipe=self.pipeline_params,
                background=self.background,
                gaussians=self.gaussians,
                q_main2vis=self.q_main2vis,
                q_vis2main=self.q_vis2main,
            )
            self.gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            self.gui_process.start()
            time.sleep(5)

    def track_update_optimizer(self, viewpoint:Camera = None, BA_flag = False) -> Optimizer:
        """
        when tracking the next frame,
        add the params of next frame into optimizer
        """
        opt_params = []
        if BA_flag:
            for cam_dix in range(len(self.current_window)):
                if self.current_window[cam_dix] == 0: # skip the first frame
                    continue
                old_viewpoint = self.cameras[self.current_window[cam_dix]]
                opt_params.append(
                    {
                        "params": [old_viewpoint.cam_rot_delta],
                        "lr": self.config["Training"]["lr"]["cam_rot_delta"] * 0.5,
                        "name": "rot_{}".format(old_viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [old_viewpoint.cam_trans_delta],
                        "lr": self.config["Training"]["lr"]["cam_trans_delta"] * 0.5,
                        "name": "trans_{}".format(old_viewpoint.uid),
                    }
                )
                opt_params.append(
                {
                    "params": [old_viewpoint.exposure_a],
                    "lr": 0.01,
                    "name": "exposure_a_{}".format(old_viewpoint.uid),
                }
                )
                opt_params.append(
                    {
                        "params": [old_viewpoint.exposure_b],
                        "lr": 0.01,
                        "name": "exposure_b_{}".format(old_viewpoint.uid),
                    }
                )
        else:
            opt_params.append(
                {
                    "params": [viewpoint.cam_rot_delta],
                    "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                    "name": "rot_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.cam_trans_delta],
                    "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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
        pose_optimizer = torch.optim.Adam(opt_params)
        return pose_optimizer
    
    def track_reset(self, cur_frame_idx, viewpoint:Camera):
        self.track_initialized = not self.monocular # High quality maps are needed
        self.keyframe_indices = []
        self.occ_aware_visibility = {}
        self.current_window = []
        self.current_window.append(cur_frame_idx)
        
        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        depth_map = self.track_get_keyframe_depth(cur_frame_idx)

        self.track_reset_flag = False 
        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                current_frame=viewpoint,
                gtcolor=viewpoint.original_image,
                gtdepth=viewpoint.depth
                if not self.monocular
                else np.zeros((viewpoint.image_height, viewpoint.image_width)),
            ))
        return depth_map
        
    def track_get_keyframe_depth(self, cur_frame_idx, depth:torch.Tensor=None, opacity:torch.Tensor=None):
        self.keyframe_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        
        valid_rgb = (gt_img.sum(dim=0) > self.rgb_boundary_threshold)[None] # valid_rgb: torch.Tensor of shape (1, H, W)
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                # depth is the render, not is gt_depth 
                depth = depth.detach().clone()
                opacity = opacity.detach().clone()
                # Update the depth with the median value for the valid pixels
                median_depth, std, valid_mask = get_median_depth(depth, opacity, mask=valid_rgb, return_std=True)
                invalid_depth_mask = torch.logical_or(depth > median_depth + std, depth < median_depth - std)
                invalid_depth_mask = torch.logical_or(invalid_depth_mask, ~valid_mask)
                depth[invalid_depth_mask] = median_depth
                # add noise to depth
                initial_depth = depth + torch.randn_like(depth) * torch.where(invalid_depth_mask, std * 0.5, std * 0.2)
                # Ignore the invalid rgb pixels
                initial_depth[~valid_rgb] = 0  
                initial_depth_numpy = initial_depth.cpu().numpy()[0]
        else:
            # use the observed depth
            initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
            initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
            initial_depth_numpy = initial_depth[0].numpy()
        return initial_depth_numpy
        
        
    def track_is_keyframe(self, cur_frame_idx, last_keyframe_idx, cur_frame_visibility_filter, occ_aware_visibility):
        # get current viewpoint and last keyframe viewpoint
        cur_viewpoint = self.cameras[cur_frame_idx]
        last_keyframe = self.cameras[last_keyframe_idx]
        # check_time
        check_full_window = len(self.current_window) >= self.window_size
        check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
        # count the distance
        pose_CW = getWorld2View2(cur_viewpoint.R, cur_viewpoint.T)
        last_kf_CW = getWorld2View2(last_keyframe.R, last_keyframe.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        # Distance check
        check1_dist = dist > self.kf_translation * self.median_depth
        check2_min_dist = dist > self.kf_min_translation * self.median_depth
        # Common visibility of Gauss points: self.kf_overlap
        union = torch.logical_or(cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]).count_nonzero()
        intersection = torch.logical_and(cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]).count_nonzero()
        point_ratio = intersection / union
        check3_visibility = point_ratio < self.kf_overlap
        if check_full_window:
            return ((check3_visibility and check2_min_dist) or check1_dist) and check_time
        else:
            return (check3_visibility and check_time)
        
        
    def track_update_keyframe_window(self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window:list):
        near_window_num = 2 # last frame and current frame still in window
        window = [cur_frame_idx] + window
        cur_viewpoint = self.cameras[cur_frame_idx]
        remove_frames_list = []
        removed_frame = None
        # Remove the farthest frame which visibility is low
        for i in range(near_window_num, len(window)): 
            keyframe_idx = window[i]
            # print(f"Checking visibility for {keyframe_idx}")
            intersection = torch.logical_and(cur_frame_visibility_filter, occ_aware_visibility[keyframe_idx]).count_nonzero()
            denom = min(cur_frame_visibility_filter.count_nonzero(), occ_aware_visibility[keyframe_idx].count_nonzero())
            point_ratio = intersection / denom
            cut_off = 0.4 if not self.track_initialized else self.cut_off
            if point_ratio <= cut_off:
                remove_frames_list.append(keyframe_idx)
        if remove_frames_list:
            window.remove(remove_frames_list[-1])
            removed_frame = remove_frames_list[-1]
        
        # If full window and all the frame is near by current frame, remove the nearest frame 
        kf_0_WC = torch.linalg.inv(getWorld2View2(cur_viewpoint.R, cur_viewpoint.T))
        if len(window) > self.window_size:
            inv_dist = []
            for i in range(near_window_num, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(near_window_num, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))
            idx = np.argmax(inv_dist)
            removed_frame = window[near_window_num + idx]
            window.remove(removed_frame)
        
        return window, removed_frame
        
    def tracking(self, cur_frame_idx, viewpoint:Camera):
        # set the prev viewpoint as init 
        prev_viewpoint = self.cameras[cur_frame_idx - 1]
        viewpoint.update_RT(prev_viewpoint.R, prev_viewpoint.T) 
        # init the optimizer
        pose_optimizer = self.track_update_optimizer(viewpoint)
        # loop to tracking pose
        for tracking_itr in range(self.tracking_itr_num):
            # not render semantic
            render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background) 
            image, depth, opacity = (render_pkg["render"], render_pkg["depth"], render_pkg["opacity"])
            # update 
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(self.config, image, depth, opacity, viewpoint)
            loss_tracking.backward()
            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint) 
            # TODO： update GUI
            if converged:
                break
        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg
    
    def map_rest(self):
        self.map_iteration_count = 0
        self.occ_aware_visibility = {} # TODO
        self.map_initialized = not self.monocular
        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
    
    def map_add_next_kf(self, frame_idx, viewpoint:Camera, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map)
    
    def map_init(self, cur_frame_idx, viewpoint:Camera):
        if self.semantic_flag:
            gt_semantic_path = self.dataset.get_pred_semantic(cur_frame_idx)
            gt_feature = torch.load(gt_semantic_path, weights_only=True).cuda()
        for mapping_iteration in range(self.init_itr_num):
            self.map_iteration_count += 1
            render_semantic_flag = self.semantic_flag and mapping_iteration > self.init_itr_num - MAPPING_START_ITR
            render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background, flag_semantic=render_semantic_flag)
            (image, viewspace_point_tensor, visibility_filter, radii, depth, opacity, n_touched, feature_map) = (
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
                l1_feature = l1_loss(feature_map, gt_feature)
                loss_init += l1_feature
                if mapping_iteration % 20 == 0:
                    print(f"Init Iteration: {mapping_iteration}, Loss: {loss_init.item()}")
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
                if self.map_iteration_count == self.init_gaussian_reset or self.map_iteration_count == self.opt_params.densify_from_iter:
                    self.gaussians.reset_opacity()
                
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.cnn_decoder_optimizer.step()
                self.cnn_decoder_optimizer.zero_grad()
        
        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()    
        Log("Initialized map")
        torch.cuda.empty_cache()
        return render_pkg
    
    def map_full_window(self, prune=False, iters=1):
        if len(self.current_window) == 0:
            return 
        
        viewpoint_stack = [self.cameras[kf_idx] for kf_idx in self.current_window]
        random_viewpoint_stack = [self.cameras[cam_idx] for cam_idx in self.keyframe_indices if cam_idx not in self.current_window]
        random_update_num = 2
        # Local BA optimizer
        pose_optimizer = self.track_update_optimizer(BA_flag=True)
        gt_feature_stack = []
        if self.semantic_flag:
            for i in range(len(self.current_window)):
                gt_semantic_path = self.dataset.get_pred_semantic(self.current_window[i])
                gt_feature = torch.load(gt_semantic_path, weights_only=True).cuda()
                gt_feature_stack.append(gt_feature)
        
        for mapping_iteration in range(iters):
            self.map_iteration_count += 1
            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []
            
            for cam_idx in range(len(self.current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                render_semantic_flag = self.semantic_flag and mapping_iteration > iters - MAPPING_ITR
                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background, flag_semantic=render_semantic_flag)
                (image, viewspace_point_tensor, visibility_filter, radii, depth, opacity, n_touched, feature_map) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                    render_pkg["feature_map"]
                )
                loss_mapping += get_loss_mapping(self.config, image, depth, viewpoint, opacity)
                if render_semantic_flag:
                    feature_map = self.cnn_decoder(F.interpolate(feature_map.unsqueeze(0), 
                                                                size=(LSeg_IMAGE_SIZE[0], LSeg_IMAGE_SIZE[1]),
                                                                mode="bilinear", align_corners=True).squeeze(0))
                    gt_feature = gt_feature_stack[cam_idx]
                    l1_feature = l1_loss(feature_map, gt_feature)
                    loss_mapping += l1_feature

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)
                
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:random_update_num]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
                (image, viewspace_point_tensor, visibility_filter, radii, depth, opacity, n_touched,) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(self.config, image, depth, viewpoint, opacity)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            
            with torch.no_grad():
                gaussian_split = False
                # update all keyframes visibility
                self.occ_aware_visibility = {}
                for idx in range(len(self.current_window)):
                    kf_idx = self.current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()
                    
                if prune:
                    if len(self.current_window) == self.window_size:
                        prune_coviz = 3 # 30%
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if self.prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < prune_coviz
                            # make sure we don't split the gaussians, break here.
                        if self.prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(self.current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2] if self.map_initialized else self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(self.gaussians.n_obs <= prune_coviz, mask)
                        
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(self.current_window))):
                                current_idx = self.current_window[idx]
                                self.occ_aware_visibility[current_idx] = self.occ_aware_visibility[current_idx][~to_prune]
                                    
                        if not self.map_initialized:
                            self.map_initialized = True
                            Log("Initialized SLAM")
                        # make sure we don't split the gaussians, break here.
                    return False
                
                # update the max radii of the gaussians
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(self.gaussians.max_radii2D[visibility_filter_acm[idx]], radii_acm[idx][visibility_filter_acm[idx]])
                    self.gaussians.add_densification_stats(viewspace_point_tensor_acm[idx], visibility_filter_acm[idx])
                update_gaussian = self.map_iteration_count % self.gaussian_update_every == self.gaussian_update_offset

                ## Gaussian update densify_and_prune
                if update_gaussian:
                    self.gaussians.densify_and_prune(self.opt_params.densify_grad_threshold,
                                                     self.gaussian_th, self.gaussian_extent, self.size_threshold)
                    gaussian_split = True
                    
                ## Opacity reset
                if self.map_iteration_count % self.gaussian_reset == 0 and not update_gaussian:
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True
                
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.map_iteration_count)
                self.cnn_decoder_optimizer.step()
                self.cnn_decoder_optimizer.zero_grad()
                pose_optimizer.step()
                pose_optimizer.zero_grad()

                
        torch.cuda.empty_cache()
        return gaussian_split

    def eval(self):
        track_metrics = Eval_Tracking(self.cameras, self.save_dir, self.monocular)
        map_metrics = Eval_Mapping(self.cameras, self.dataset, 
                                   self.gaussians, self.pipeline_params, self.background,
                                   self.save_dir, self.monocular, interval=1)
        return {"track_metrics": track_metrics, "map_metrics": map_metrics}
    
    def eval_keyframes(self):
        keyframes = {}
        for kf_idx in self.keyframe_indices:
            keyframes[kf_idx] = self.cameras[kf_idx]
        track_metrics = Eval_Tracking(keyframes, monocular=self.monocular)
        map_metrics = Eval_Mapping(keyframes, self.dataset, 
                                   self.gaussians, self.pipeline_params, self.background,
                                   monocular=self.monocular, interval=1)
        semantic_metrics = Eval_Semantic(keyframes, self.dataset, 
                                   self.gaussians, self.pipeline_params, self.background,
                                   monocular=self.monocular, interval=1)
        kf_output = {"track_metrics": track_metrics, "map_metrics": map_metrics, "semantic_metrics": semantic_metrics}
        with open(os.path.join(self.save_dir, f"eval_kf_{self.keyframe_indices[-1]:06}.json"), 'w', encoding='utf-8') as f:
            json.dump(kf_output, f, indent=4)
        return kf_output
    
    def resume_gui(self):
        self.gaussians.load_ply("/home/MonoGS_Semantic/checkpoints/gaussian_kf_001864.ply")
        cnn_ckpts = torch.load("/home/MonoGS_Semantic/checkpoints/cnn_decoder_dim64.pth", weights_only=True)
        self.gaussians.semantic_decoder = cnn_ckpts
        self.run_gui()
        cur_frame_idx = 0
        self.current_window.append(cur_frame_idx)
        viewpoint = Camera.init_from_dataset(self.dataset, cur_frame_idx, self.projection_matrix)
        viewpoint.compute_grad_mask(self.config)
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        self.cameras[cur_frame_idx] = viewpoint
        keyframes = [Camera.copy_camera(self.cameras[kf_idx]) for kf_idx in self.current_window]
        # current_window_dict = {}
        # current_window_dict[self.current_window[0]] = self.current_window[1:]
        while True:
            time.sleep(1)
            viewpoint = Camera.init_from_dataset(self.dataset, cur_frame_idx, self.projection_matrix)
            self.q_main2vis.put(gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians),
                                                        gtcolor=viewpoint.original_image,
                                                        gtdepth=viewpoint.depth
                                                        if not self.monocular
                                                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                                                        current_frame=viewpoint,
                                                        keyframes=keyframes,
                                                        # kf_window=current_window_dict,
                                                         ))
            
    def resume_run(self, gaussians_ckpt, cnn_ckpt, resume_info):
        self.gaussians.load_ply(gaussians_ckpt)
        cnn_ckpt_dict = torch.load(cnn_ckpt, weights_only=True)
        self.cnn_decoder.load_state_dict(cnn_ckpt_dict)
        self.gaussians.semantic_decoder = cnn_ckpt_dict
        print(self.gaussians._xyz.shape)
        
        self.keyframe_indices = resume_info["keyframe_indices"]
        self.current_window = resume_info["current_window"]
        print("resume current_window", self.current_window)
        print("resume keyframe_indices", self.keyframe_indices)
        frames_pose_dict = resume_info["pose_dict"]
        self.cameras = {}
        self.occ_aware_visibility = {}
        max_idx = self.keyframe_indices[-1]
        for id in range(max_idx+1):
            viewpoint = Camera.init_from_dataset(self.dataset, id, self.projection_matrix)
            viewpoint.compute_grad_mask(self.config)
            self.cameras[id] = viewpoint
            cur_R = torch.tensor(frames_pose_dict[str(id)]['R'], dtype=torch.float32, device="cuda")
            cur_T = torch.tensor(frames_pose_dict[str(id)]['T'], dtype=torch.float32, device="cuda")
            self.cameras[id].R = cur_R
            self.cameras[id].T = cur_T
            self.cameras[id].update_RT(cur_R, cur_T)
            if id not in self.keyframe_indices:
                self.cameras[id].clean() 
                
        for kf_idx in self.keyframe_indices:
            viewpoint = self.cameras[kf_idx]
            render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
            cur_occ_aware_visibility = (render_pkg["n_touched"] > 0).long()
            self.occ_aware_visibility[kf_idx] = cur_occ_aware_visibility

        return self.keyframe_indices[-1] + 1
        
    
    def save_state(self, cur_frame_idx):
        self.gaussians.save_ply(path=os.path.join(self.save_dir, f"gaussian_kf_{cur_frame_idx:06}.ply"))
        decoder_state_dict = self.cnn_decoder.state_dict()
        torch.save(decoder_state_dict, os.path.join(self.save_dir, f"decoder_{cur_frame_idx:06}.pth"))
        
        pose_dict = {}
        for idx, viewpoint in self.cameras.items():
            pose = {"R": viewpoint.R.cpu().numpy().tolist(), "T": viewpoint.T.cpu().numpy().tolist()}
            pose_dict[idx] = pose
        
        resume_info = {"current_window": self.current_window, "keyframe_indices": self.keyframe_indices, "pose_dict": pose_dict}
        with open(os.path.join(self.save_dir, f"resume_info_{cur_frame_idx:06}.json"), 'w', encoding='utf-8') as f:
            json.dump(resume_info, f, indent=4)
    
    def update_gaussians_decoder(self):
        decoder_state_dict = self.cnn_decoder.state_dict()
        state_dict_cpu = {key: value.cpu() for key, value in decoder_state_dict.items()}
        self.gaussians.semantic_decoder = state_dict_cpu
    
    def run(self, resume=None):
        cur_frame_idx = 0
        resume_flag = True
        self.run_gui()
        while True:
            if cur_frame_idx > len(self.dataset) - 1:
                self.eval()
                break
            elif len(self.keyframe_indices) % self.save_trj_kf_intv == 0 \
                    and len(self.keyframe_indices) and self.keyframe_indices[-1] == cur_frame_idx-1:
                self.save_state(cur_frame_idx)
                # self.gaussians.save_ply(path=os.path.join(self.save_dir, f"kf_{cur_frame_idx}.ply"))
                # decoder_state_dict = self.cnn_decoder.state_dict()
                # torch.save(decoder_state_dict, os.path.join(self.save_dir, f"decoder_{cur_frame_idx}.pth"))
                output = self.eval_keyframes()
                print(output) 
                
            if resume and resume_flag:
                gaussians_ckpt = sorted(glob.glob(f"{resume}/gaussian_kf_*.ply"))[-1]
                cnn_ckpt = sorted(glob.glob(f"{resume}/decoder_*.pth"))[-1]
                resume_info = json.load(open(sorted(glob.glob(f"{resume}/resume_info_*.json"))[-1], 'r'))
                self.track_reset_flag = False
                self.semantic_init()
                cur_frame_idx = self.resume_run(gaussians_ckpt, cnn_ckpt, resume_info)
                resume_flag = False
                
            # get the current frame
            viewpoint = Camera.init_from_dataset(self.dataset, cur_frame_idx, self.projection_matrix)
            viewpoint.compute_grad_mask(self.config)
            self.cameras[cur_frame_idx] = viewpoint
            
            # reset the track and map
            if self.track_reset_flag:
                print("reset")
                self.semantic_init()
                depth_map = self.track_reset(cur_frame_idx, viewpoint)
                self.map_rest()
                self.map_add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, init=True)
                self.map_init(cur_frame_idx, viewpoint)
                self.update_gaussians_decoder()
                cur_frame_idx += 1
                continue
            
            # If window full, track is initialized.
            self.track_initialized = self.track_initialized or len(self.current_window) >= self.window_size
            
            # tracking
            track_start_time = time.time()
            render_pkg = self.tracking(cur_frame_idx, viewpoint)
            # ate_output = Eval_frame_pose(viewpoint, monocular=self.monocular)
            print(f"[{cur_frame_idx}] track time: {time.time()-track_start_time}")
            
            current_window_dict = {}
            current_window_dict[self.current_window[0]] = self.current_window[1:]
            keyframes = [Camera.copy_camera(self.cameras[kf_idx]) for kf_idx in self.current_window]

            self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )
            
            # check keyframe
            last_keyframe_idx = self.current_window[0]
            check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
            curr_visibility = (render_pkg["n_touched"] > 0).long()
            
            create_kf = self.track_is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
            if len(self.current_window) < self.window_size:
                union = torch.logical_or(curr_visibility, self.occ_aware_visibility[last_keyframe_idx]).count_nonzero()
                intersection = torch.logical_and(curr_visibility, self.occ_aware_visibility[last_keyframe_idx]).count_nonzero()
                point_ratio = intersection / union
                create_kf = (check_time and point_ratio < self.config["Training"]["kf_overlap"])
            
            if create_kf and check_time:
                self.current_window, removed = self.track_update_keyframe_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                if self.monocular and not self.track_initialized and removed is not None:
                    self.track_reset = True
                    Log(
                        "Keyframes lacks sufficient overlap to initialize the map, resetting."
                    )
                    continue
                depth_map = self.track_get_keyframe_depth(cur_frame_idx, depth=render_pkg["depth"], opacity=render_pkg["opacity"])
                # mapping
                map_start_time = time.time()
                print("--------------")
                print(self.gaussians._xyz.shape)
                self.map_add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)
                print("--------------")
                print(self.gaussians._xyz.shape)
                self.map_full_window(iters=self.mapping_itr_num)
                self.map_full_window(prune=True)
                self.update_gaussians_decoder()
                print(f"[{cur_frame_idx}] map time: {time.time()-map_start_time} keyframes_num: {len(self.keyframe_indices)} map_window:{self.current_window}")
            else:
                # delete the frame
                self.cameras[cur_frame_idx].clean() 
                torch.cuda.empty_cache()
            cur_frame_idx += 1 
                

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    # if args.eval:
    #     Log("Running MonoGS in Evaluation Mode")
    #     Log("Following config will be overriden")
    #     Log("\tsave_results=True")
    #     config["Results"]["save_results"] = True
    #     Log("\tuse_gui=False")
    #     config["Results"]["use_gui"] = False
    #     Log("\teval_rendering=True")
    #     config["Results"]["eval_rendering"] = True
    #     Log("\tuse_wandb=True")
    #     config["Results"]["use_wandb"] = True
    
    if not args.eval:
        if config["Results"]["save_results"]:
            mkdir_p(config["Results"]["save_dir"])
            current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            path = config["Dataset"]["dataset_path"].split("/")
            save_dir = os.path.join(
                config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
            )
            tmp = args.config
            tmp = tmp.split(".")[0]
            config["Results"]["save_dir"] = save_dir
            mkdir_p(save_dir)
            with open(os.path.join(save_dir, "config.yml"), "w") as file:
                documents = yaml.dump(config, file)
            Log("saving results in " + save_dir)

    slam = SLAM(config, save_dir=save_dir)

    try:
        if args.eval:
            slam.resume_gui()
        else:
            slam.run(resume=args.resume)
    except KeyboardInterrupt:
        Log("KeyboardInterrupt. Exiting GUI...")
        slam.gui_process.close()
    finally:
        Log("Exiting GUI...")
        slam.gui_process.close()

