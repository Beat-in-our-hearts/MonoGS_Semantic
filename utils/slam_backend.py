import os
import random
import sys
import time
from typing import Dict, List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log, debug, info
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping

from utils.camera_utils import Camera
from utils.semantic_utils import build_decoder, label_loss, create_dense_feature
from utils.semantic_setting import Semantic_Config

from diff_gaussian_rasterization import get_semantic_channels

class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians:GaussianModel = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False
        self.dataset = None

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints:Dict[int, Camera] = dict()
        if Semantic_Config.preload_semantic:
            self.gt_semantic_stack:Dict[int, torch.Tensor] = dict()
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        
        # CNN Decoder to upsample semantic features
        if Semantic_Config.enable and Semantic_Config.mode in ["SAM2", "CLIP", "SAM_CLIP", "Grounding_Dino"]:
            self.cnn_decoder, self.cnn_decoder_optimizer = build_decoder()
            if Semantic_Config.mode == "Grounding_Dino":
                self.cnn_decoder.load_state_dict(torch.load("checkpoints/decoder_128_512.pth"))
                
        self.gt_text_features = None

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

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def track_update_optimizer(self, viewpoint:Camera = None, BA_flag = False, GBA_flag = False):
        """
        when tracking the next frame,
        add the params of next frame into optimizer
        """
        opt_params = []

        if BA_flag:
            if GBA_flag and len(self.current_window) == self.window_size:
                frames_to_optimize = self.window_size - 1
            else:
                frames_to_optimize = self.config["Training"]["pose_window"]
            for cam_dix in range(min(frames_to_optimize, len(self.current_window))):
                if self.current_window[cam_dix] == 0: # skip the first frame
                    continue
                old_viewpoint = self.viewpoints[self.current_window[cam_dix]]
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
        elif viewpoint is not None:
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
    
    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            
            render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(self.config, image, depth, viewpoint, opacity, initialization=True)
            loss_init.backward()
            
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        torch.cuda.empty_cache()
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
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
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
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
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                if self.keyframe_optimizers is not None:
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
            torch.cuda.empty_cache()
        return gaussian_split

    def map_grounding_dino(self, iters=1, window_size=2, init_flag=False):
        if Semantic_Config.mode in ["Grounding_Dino_v2", "Grounding_Dino"] and Semantic_Config.enable:
            raise ValueError('''Semantic_Config.mode in ['Grounding_Dino_v2', 'Grounding_Dino'] and Semantic_Config.enable''')

        if len(self.current_window) == 0:
            return
        # TODO random select previous frame in semantic_window
        semantic_window = self.current_window[:window_size]
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in semantic_window]
        
        render_pkg_mask_stack = {}
        
        with torch.no_grad():
            for cam_idx in semantic_window:
                if cam_idx == 0:
                
                
    def map_semantic(self, iters=1, window_size=2, init_flag=False):
        start_time = time.time()
        if not Semantic_Config.enable:
            return
        
        if len(self.current_window) == 0:
            return
        
        semantic_window = self.current_window[:window_size]
        
        # NOTE random select frames to add into the semantic window
        if Semantic_Config.Semantic_Debug["random_select"]:
            if len(self.viewpoints) > 4:
                random_idx_stack = []
                for cam_idx, viewpoint in self.viewpoints.items():
                    if cam_idx in semantic_window:
                        continue
                    random_idx_stack.append(cam_idx)
                random_select_num = 1
                semantic_window = semantic_window + random.sample(random_idx_stack, random_select_num)
                
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in semantic_window]
        
        # NOTE rerender the semantic feature
        with torch.no_grad():
            if Semantic_Config.mode in ["Grounding_Dino_v2", "Grounding_Dino"]:
                render_pkg_mask_stack = []
                for cam_idx in range(len(semantic_window)):
                    viewpoint = viewpoint_stack[cam_idx]
                    render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background, 
                                        flag_semantic=True)
                    feature_map = render_pkg["feature_map"]
                    feature_map = self.cnn_decoder(feature_map)
                    pred_ssim = feature_map.permute(1, 2, 0) @ self.gt_text_features.T
                    threshold = 0.2
                    black_mask = (pred_ssim < threshold).all(dim=-1)
                    render_pkg_mask_stack.append(black_mask.detach()) # black mask
         
        tensor_label_stack = []
        sparse_mask_stack = []
        pred_feature_stack = []
        lseg_label_stack = []
        if Semantic_Config.mode == "GT_Label": # GT label
            if Semantic_Config.GT_Exp["mode"] == "Sparse GT":
                # define sparse GT label path
                sparse_label_dir = os.path.join(self.config["Results"]["save_dir"], "sparse_gt", "label")
                sparse_mask_dir = os.path.join(self.config["Results"]["save_dir"], "sparse_gt", "mask")
            
                if init_flag:
                    Log(f"Sparse GT: {Semantic_Config.GT_Exp['sparse_ratio']}")
                
                    os.makedirs(sparse_label_dir, exist_ok=True)
                    os.makedirs(sparse_mask_dir, exist_ok=True)
                        
                    # generate sparse GT label
                    for i in tqdm(range(self.dataset.num_imgs)):
                        gt_label_path = self.dataset.get_gt_semantic(i)
                        label_img = cv2.imread(gt_label_path, cv2.IMREAD_GRAYSCALE)
                        
                        pix_W, pix_H = label_img.shape
                        num_pix = pix_W * pix_H
                        num_keep = int(num_pix * Semantic_Config.GT_Exp["sparse_ratio"])
                        indices_keep = np.random.choice(num_pix, num_keep, replace=False)
                        
                        temp_label = np.zeros_like(label_img)
                        mask = np.zeros_like(label_img, dtype=bool)
                        temp_label.flat[indices_keep] = label_img.flat[indices_keep]
                        mask.flat[indices_keep] = True
                        
                        cv2.imwrite(os.path.join(sparse_label_dir, f"{i:04d}.png"), temp_label)
                        cv2.imwrite(os.path.join(sparse_mask_dir, f"{i:04d}.png"), (mask*255).astype("uint8"))
            
                for i in range(len(semantic_window)):
                    sparse_label_path = os.path.join(sparse_label_dir, f"{semantic_window[i]:04d}.png")
                    sparse_mask_path = os.path.join(sparse_mask_dir, f"{semantic_window[i]:04d}.png")
                    label_img = cv2.imread(sparse_label_path, cv2.IMREAD_GRAYSCALE)
                    sparse_mask_img = cv2.imread(sparse_mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    gt_label = torch.tensor(label_img).long().cuda()
                    tensor_label_stack.append(gt_label.detach())
                    sparse_mask = torch.tensor(sparse_mask_img).long().cuda().to(torch.bool)
                    sparse_mask_stack.append(sparse_mask.detach())
            elif Semantic_Config.GT_Exp["mode"] == "Noise GT":
                # define noise GT label path
                noise_label_dir = os.path.join(self.config["Results"]["save_dir"], "noise_gt", "label")
                
                if init_flag:
                    Log(f"Noise GT: {Semantic_Config.GT_Exp['noise_ratio']}")
                    
                    os.makedirs(noise_label_dir, exist_ok=True)
                    # generate noise GT label
                    for i in tqdm(range(self.dataset.num_imgs)):
                        gt_label_path = self.dataset.get_gt_semantic(i)
                        label_img = cv2.imread(gt_label_path, cv2.IMREAD_GRAYSCALE)
                        
                        pix_W, pix_H = label_img.shape
                        num_pix = pix_W * pix_H
                        num_noise = int(num_pix * Semantic_Config.GT_Exp["noise_ratio"])
                        indices_noise = np.random.choice(num_pix, num_noise, replace=False)
                        
                        # combine noise label with GT label
                        temp_label = np.zeros_like(label_img)
                        temp_label.flat[indices_noise] = np.random.randint(0, get_semantic_channels(), num_noise)
                        temp_label = np.where(temp_label == 0, label_img, temp_label)
                        
                        cv2.imwrite(os.path.join(noise_label_dir, f"{i:04d}.png"), temp_label)
                
                for i in range(len(semantic_window)):
                    noise_label_path = os.path.join(noise_label_dir, f"{semantic_window[i]:04d}.png")
                    label_img = cv2.imread(noise_label_path, cv2.IMREAD_GRAYSCALE)
                    gt_label = torch.tensor(label_img).long().cuda()
                    tensor_label_stack.append(gt_label.detach())        
                
            else: # Normal GT label
                for i in range(len(semantic_window)):
                    gt_label_path = self.dataset.get_gt_semantic(semantic_window[i])
                    label_img = cv2.imread(gt_label_path, cv2.IMREAD_GRAYSCALE) # W H
                    gt_label = torch.tensor(label_img).long().cuda()
                    tensor_label_stack.append(gt_label.detach())
        elif Semantic_Config.mode in ["SAM_CLIP", "Grounding_Dino"]: # pred label with feature
            for i in range(len(semantic_window)):
                pred_label_path = self.dataset.get_pred_label(semantic_window[i])
                label_img = cv2.imread(pred_label_path, cv2.IMREAD_GRAYSCALE)
                pred_label = torch.tensor(label_img).long().cuda()
                tensor_label_stack.append(pred_label)
                
                pred_semantic_path = self.dataset.get_pred_semantic(semantic_window[i])
                pred_feature = torch.load(pred_semantic_path, weights_only=True) 
                pred_feature_stack.append(pred_feature)
                
                # NOTE lseg label 360 x 480
                if Semantic_Config.use_lseg:
                    pred_lseg_label_path = self.dataset.get_pred_lseg_label(semantic_window[i])
                    lseg_label_img = cv2.imread(pred_lseg_label_path, cv2.IMREAD_GRAYSCALE)
                    lseg_label = torch.tensor(lseg_label_img).long().cuda()
                    lseg_label_stack.append(lseg_label)
        elif Semantic_Config.mode in ["Grounding_Dino_v2"]:
            for i in range(len(semantic_window)):
                pred_semantic_path = self.dataset.get_pred_semantic(semantic_window[i])
                pred_feature = torch.load(pred_semantic_path, weights_only=True) 
                if pred_feature["class_names"] != None:
                    pred_feature_stack.append(pred_feature)
                    
                    pred_label_path = self.dataset.get_pred_label(semantic_window[i])
                    label_img = cv2.imread(pred_label_path, cv2.IMREAD_GRAYSCALE)
                    pred_label = torch.tensor(label_img).long().cuda()
                    tensor_label_stack.append(pred_label)    
                else:
                    pred_feature_stack.append(None)
                    tensor_label_stack.append(None)
        else:
            if Semantic_Config.preload_semantic:
                for i in range(len(semantic_window)):
                    pred_feature_stack.append(self.gt_semantic_stack[semantic_window[i]])
            else:
                for i in range(len(semantic_window)):
                    pred_semantic_path = self.dataset.get_pred_semantic(semantic_window[i])
                    pred_feature = torch.load(pred_semantic_path, weights_only=True).cuda()
                    pred_feature_stack.append(pred_feature)
    
        semantic_loss = []
        for _ in range(iters):
            loss_semantic = 0
            for cam_idx in range(len(semantic_window)):
                viewpoint = viewpoint_stack[cam_idx]
                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background,
                                    flag_semantic=True)
                feature_map = render_pkg["feature_map"]
                if Semantic_Config.mode in ["SAM2", "CLIP"]:
                    render_size = Semantic_Config.render_size
                    feature_map = self.cnn_decoder(F.interpolate(feature_map.unsqueeze(0), render_size,
                                                                mode="bilinear", align_corners=True).squeeze(0))
                    pred_feature = pred_feature_stack[cam_idx]
                    pred_feature = F.interpolate(pred_feature.unsqueeze(0), render_size, mode="bilinear", align_corners=True).squeeze(0)
                    l1_feature = l1_loss(feature_map, pred_feature)
                    loss_semantic += l1_feature
                elif Semantic_Config.mode in ["Grounding_Dino_v2"]:
                    # no decect
                    if pred_feature_stack[cam_idx] == None and tensor_label_stack[cam_idx] == None: 
                        continue
                    render_size = Semantic_Config.render_size
                    feature_map = self.cnn_decoder(F.interpolate(feature_map.unsqueeze(0), render_size,mode="bilinear", align_corners=True).squeeze(0))
                    
                    # resize the supervise signal
                    pred_label = tensor_label_stack[cam_idx]  
                    pred_feature = pred_feature_stack[cam_idx]
                    pred_dense_feature = create_dense_feature(pred_label, pred_feature, Semantic_Config.semantic_dim[Semantic_Config.mode])
                    pred_dense_feature = F.interpolate(pred_dense_feature.unsqueeze(0), render_size, mode="bilinear", align_corners=True).squeeze(0)
                    
                    # resize the supervise mask
                    mask = (pred_label != 0).float().unsqueeze(0) 
                    mask = F.interpolate(mask.unsqueeze(0), render_size, mode="bilinear", align_corners=True).squeeze(0).squeeze(0).to(torch.bool)
                    
                    
                     
                elif Semantic_Config.mode == "GT_Label":
                    if Semantic_Config.GT_Exp["mode"] == "Sparse GT":
                        gt_label = tensor_label_stack[cam_idx]
                        mask = sparse_mask_stack[cam_idx]
                        
                        pred_feature = feature_map[:,mask].unsqueeze(0)
                        gt_label = gt_label[mask].unsqueeze(0)
                        
                        loss_label = label_loss(pred_feature, gt_label)
                        loss_semantic += loss_label
                    else:
                        gt_label = tensor_label_stack[cam_idx]
                        loss_label = label_loss(feature_map.unsqueeze(0), gt_label.unsqueeze(0))
                        loss_semantic += loss_label
                elif Semantic_Config.mode in ["SAM_CLIP", "Grounding_Dino"]:
                    # resize the feature map
                    render_size = Semantic_Config.render_size
                    feature_map = self.cnn_decoder(F.interpolate(feature_map.unsqueeze(0), render_size,
                                                                mode="bilinear", align_corners=True).squeeze(0))
                    # resize the pred label and feature
                    pred_label = tensor_label_stack[cam_idx]
                    pred_feature = pred_feature_stack[cam_idx]

                    pred_dense_feature = create_dense_feature(pred_label, pred_feature, 
                                                              Semantic_Config.semantic_dim[Semantic_Config.mode])
                    pred_dense_feature = F.interpolate(pred_dense_feature.unsqueeze(0), render_size, mode="bilinear", align_corners=True).squeeze(0)
                    
                    mask = (pred_label != 0).float().unsqueeze(0) # NOTE pred vaild mask
                    mask = F.interpolate(mask.unsqueeze(0), render_size, mode="bilinear", align_corners=True).squeeze(0).squeeze(0).to(torch.bool)
                    
                    # TODO rerender the prev fmap, fix mask
                    if len(render_pkg_mask_stack) != 0:
                        pred_empty_mask = ~mask
                        rerender_empty_mask = render_pkg_mask_stack[cam_idx]
                        rerender_empty_mask = rerender_empty_mask.float().unsqueeze(0)
                        rerender_empty_mask = F.interpolate(rerender_empty_mask.unsqueeze(0), render_size, mode="bilinear", align_corners=True).squeeze(0).squeeze(0).to(torch.bool)
                        empty_mask = torch.logical_and(pred_empty_mask, rerender_empty_mask)
                        
                        # fill the mini hole of empty_mask, use Erosion than Dilation
                        empty_mask_unsqueeze = empty_mask.unsqueeze(0).unsqueeze(0).float()
                        max_pool = nn.MaxPool2d(kernel_size=9, stride=1, padding=4) # 5x5 kernel
                        empty_mask_unsqueeze = -max_pool(empty_mask_unsqueeze) # Erosion
                        empty_mask_unsqueeze = max_pool(empty_mask_unsqueeze) # Dilation
                        empty_mask = empty_mask_unsqueeze.squeeze(0).squeeze(0).to(torch.bool)
                        
                        mask = torch.logical_or(mask, empty_mask) # mask = vaild_mask + empty_mask
                    
                    # TODO fussion lseg feature for wall and floor
                    # force overwrite
                    if Semantic_Config.use_lseg:
                        pred_dense_feature = pred_dense_feature.permute(1, 2, 0)
                        lseg_mask = torch.zeros(pred_dense_feature.shape[:2], dtype=torch.bool).cuda()
                        pred_lseg_label = lseg_label_stack[cam_idx]
                        pred_lseg_word_dict = pred_feature["sp_word_features"] # {"wall": .., "floor": ..}
                        word_info_dict = {
                            "wall": {"word_id":1, "force_mask":False},
                            "floor": {"word_id":4, "force_mask":True},
                            "lamp": {"word_id":37, "force_mask":True},
                            "ceiling": {"word_id":6, "force_mask":False},
                        }
                        for word, word_feature in pred_lseg_word_dict.items():
                            word_info = word_info_dict[word]
                            word_id = word_info["word_id"]
                            force_mask_enable = word_info["force_mask"]
                            if force_mask_enable:
                                force_mask = pred_lseg_label == word_id
                                pred_dense_feature[force_mask] = word_feature
                                lseg_mask = torch.logical_or(lseg_mask, force_mask)
                            else:
                                soft_mask = (pred_lseg_label == word_id) & (~mask)
                                pred_dense_feature[soft_mask] = word_feature
                                lseg_mask = torch.logical_or(lseg_mask, soft_mask)
                        pred_dense_feature = pred_dense_feature.permute(2, 0, 1)
                        mask = torch.logical_or(mask, lseg_mask)
                    
                    cv2.imwrite(f"results/test/vis/mask_{cam_idx:04d}.png", (mask*255).detach().cpu().numpy().astype("uint8"))
                    
                    # Create a mask to ignore background (label=0) regions
                    pred_dense_feature = pred_dense_feature.detach() # no grad in pred_dense_feature
                    mask = mask.detach()
                    if not init_flag:
                        feature_map[:, ~mask] = 0
                        pred_dense_feature[:, ~mask] = 0
                    l1_feature = l1_loss(feature_map, pred_dense_feature)
                    loss_semantic += l1_feature
                else:
                    raise NotImplementedError
            semantic_loss.append(loss_semantic.item())
            if len(semantic_loss) % 10 == 0:
                eval_loss = semantic_loss[-10:]
                Log(f"semantic loss: {sum(eval_loss)/10/len(semantic_window)}")
            loss_semantic.backward()
            with torch.no_grad():
                # if Semantic_Config.mode in ["SAM2", "CLIP", "SAM_CLIP", "Grounding_Dino"]:
                #     self.cnn_decoder_optimizer.step()
                #     self.cnn_decoder_optimizer.zero_grad()
                self.gaussians.semantic_optimizer.step()
                self.gaussians.semantic_optimizer.zero_grad()
            
        debug(f"semantic mapping time: {time.time()-start_time:.1f}")
                
    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1), mininterval=5):
            loss = 0
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            gt_depth = torch.tensor(viewpoint_cam.depth).cuda()
            Ll1 = l1_loss(image, gt_image)
            loss += (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            if not self.monocular:
                loss += l1_loss(depth, gt_depth)
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"
        state_dict_cpu = None
        if Semantic_Config.enable:
            if Semantic_Config.mode in ["SAM2", "CLIP", "SAM_CLIP", "Grounding_Dino"]:
                decoder_state_dict = self.cnn_decoder.state_dict()
                state_dict_cpu = {key: value.cpu() for key, value in decoder_state_dict.items()}
        msg = [tag, self.gaussians.get_state_dict(), self.occ_aware_visibility, keyframes, state_dict_cpu]
        self.frontend_queue.put(msg)

    def run(self):
        torch.set_num_threads(2)
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.1)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.1)
                    continue
                if self.single_thread:
                    time.sleep(0.1)
                    continue
                self.last_sent += 1
                self.map(self.current_window)
                if self.last_sent % 10 == 0:
                    self.map(self.current_window, prune=True, iters=10)
                    debug(f"idle mapping")
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
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()
                    self.viewpoints[cur_frame_idx] = viewpoint
                    if Semantic_Config.preload_semantic:
                        gt_semantic_path = self.dataset.get_pred_semantic(cur_frame_idx)
                        gt_feature = torch.load(gt_semantic_path, weights_only=True).cuda()
                        self.gt_semantic_stack[cur_frame_idx] = gt_feature
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, init=True)
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.current_window = [cur_frame_idx]
                    self.map_semantic(iters=Semantic_Config.semantic_init_iter, init_flag=True)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    map_start_time = time.time()
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    if Semantic_Config.preload_semantic:
                        gt_semantic_path = self.dataset.get_pred_semantic(cur_frame_idx)
                        gt_feature = torch.load(gt_semantic_path, weights_only=True).cuda()
                        self.gt_semantic_stack[cur_frame_idx] = gt_feature
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    GBA_flag = False
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 20
                    if not self.initialized:
                        if len(self.current_window) == self.window_size:
                            GBA_flag = True
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num

                    self.keyframe_optimizers = self.track_update_optimizer(BA_flag=Semantic_Config.Pose_BA_flag, GBA_flag=GBA_flag)
                    
                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.map_semantic(iters=Semantic_Config.semantic_iter, window_size=Semantic_Config.semantic_window)
                    self.push_to_frontend("keyframe")
                    info(f"[{cur_frame_idx:04d}] map time: {time.time()-map_start_time:.1f} keyframes_num: {len(self.viewpoints)} map_window:{self.current_window}")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
