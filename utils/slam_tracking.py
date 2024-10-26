import os
import shutil
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np

from PIL import Image
import time
from typing import Dict, List, Union
from torch.optim.optimizer import Optimizer

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj, FakeQueue
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from gui import gui_utils

from utils.common_var import *
from feature_encoder.lseg_encoder.feature_extractor import LSeg_FeatureExtractor

class FrontEnd_Track(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = None
        self.background = None
        self.pipeline_params = None
        self.frontend_queue:mp.Queue = None
        self.backend_queue:mp.Queue = None
        self.q_main2vis:Union[mp.Queue, FakeQueue] = None
        self.q_vis2main:Union[mp.Queue, FakeQueue] = None
        

        self.initialized = False
        self.keyframe_indices = []
        self.monocular = config["Training"]["monocular"]

        self.occ_aware_visibility = {}
        self.current_window:List[int] = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0

        self.gaussians:GaussianModel = None
        self.cameras:Dict[int, Camera] = dict()
        self.device = "cuda:0"
        self.pause = False
        
        
        self.semantic_flag = False
        self.decoder_init = False
        self.save_semantic = True
        self.save_frame_visualize = True
        
        
    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
        
        self.rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        
        # keyframe check
        self.kf_translation = self.config["Training"]["kf_translation"]
        self.kf_min_translation = self.config["Training"]["kf_min_translation"]
        self.kf_overlap = self.config["Training"]["kf_overlap"]
        
        # window update
        self.cut_off = (self.config["Training"]["kf_cutoff"] if "kf_cutoff" in self.config["Training"] else 0.4)
        
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
        self.projection_matrix = self.projection_matrix.to(device=self.device)
        
    def set_feature_extractor(self):
        if self.semantic_flag:
            if self.save_semantic:
                self.save_semantic_dir = "results/temp"
                shutil.rmtree(self.save_semantic_dir)
                os.makedirs(self.save_semantic_dir)
            if self.save_frame_visualize:
                self.save_vis_semantic_dir = "results/vis"
                shutil.rmtree(self.save_vis_semantic_dir)
                os.makedirs(self.save_vis_semantic_dir)
            # Feature Extractor
            self.feature_extractor = LSeg_FeatureExtractor(debug=True)
            self.feature_extractor.eval()
            
            # CNN Decoder to upsample semantic features
            self.decoder_init = True
            self.cnn_decoder = nn.Conv2d(SEMANTIC_FEATURES_DIM, LSeg_FEATURES_DIM, kernel_size=1).to(self.device)
            self.cnn_decoder.eval() # no gradient
    
    def update_gui(self, viewpoint:Camera, update_gaussians:bool=False, update_keyframes:bool=False, vis_semantic=None):
        if update_keyframes:
            current_window_dict = {}
            current_window_dict[self.current_window[0]] = self.current_window[1:]
            # copy the keyframes without rgb, depth, semantic feature
            keyframes = [Camera.copy_camera(self.cameras[kf_idx]) for kf_idx in self.current_window]
            self.q_main2vis.put( 
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians) if update_gaussians else None,
                    current_frame=Camera.copy_camera(viewpoint),
                    keyframes=keyframes, 
                    kf_window=current_window_dict,
                    gtcolor=gui_utils.Rgb2Numpy(viewpoint.original_image),
                    gtdepth=viewpoint.depth
                    if not self.monocular
                    else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    vis_semantic=vis_semantic,
                )) 
        else:
            self.q_main2vis.put( 
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians) if update_gaussians else None,
                    current_frame=Camera.copy_camera(viewpoint),
                    gtcolor=gui_utils.Rgb2Numpy(viewpoint.original_image),
                    gtdepth=viewpoint.depth
                    if not self.monocular
                    else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    vis_semantic=vis_semantic,
                )) 
    
    def initialize_track(self, cur_frame_idx, viewpoint:Camera):
        self.initialized = not self.monocular # High quality maps are needed
        self.keyframe_indices = []
        self.occ_aware_visibility = {}
        self.current_window = []
        self.current_window.append(cur_frame_idx)
        
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        depth_map = self.get_keyframe_depth(cur_frame_idx)
        self.request_backend("init", cur_frame_idx, viewpoint, self.current_window, depth_map)
        self.reset = False 
        
        # update gt in gui
        self.update_gui(viewpoint)
        
    def update_track_optimizer(self, viewpoint:Camera) -> Optimizer:
        """
        when tracking the next frame,
        add the params of next frame into optimizer
        """
        opt_params = []
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
    
    def tracking(self, cur_frame_idx, viewpoint:Camera):
        # set the prev viewpoint as init 
        prev_viewpoint = self.cameras[cur_frame_idx - 1]
        viewpoint.update_RT(prev_viewpoint.R, prev_viewpoint.T) 
        # init the optimizer
        pose_optimizer = self.update_track_optimizer(viewpoint)
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
    
    def update_gui_visualize(self, cur_frame_idx, viewpoint:Camera):
        visualize_check = self.decoder_init and self.semantic_flag
        if visualize_check:
            render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background, flag_semantic=True)
            feature_map = render_pkg["feature_map"]
            feature_map = self.cnn_decoder(F.interpolate(feature_map.unsqueeze(0), 
                                                         size=(LSeg_IMAGE_SIZE[0], LSeg_IMAGE_SIZE[1]),
                                                         mode="bilinear", align_corners=True).squeeze(0))
            vis_feature, _ = self.feature_extractor.features_to_image(feature_map)
            if self.save_frame_visualize:
                image_save_path = os.path.join(self.save_vis_semantic_dir, f"vis_feature_{cur_frame_idx}.png")
                image = Image.fromarray(vis_feature)
                image.save(image_save_path)
        # update gt and render semantic in gui
        self.update_gui(viewpoint, update_gaussians=True, update_keyframes=True,
                        vis_semantic=vis_feature if visualize_check else None)
                
    
    def is_keyframe(self, cur_frame_idx, last_keyframe_idx, cur_frame_visibility_filter, occ_aware_visibility):
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
            if self.single_thread: # when single threa, keyframe_idx must > kf_interval
                return ((check3_visibility and check2_min_dist) or check1_dist) and check_time
            else:
                return (check3_visibility and check2_min_dist) or check1_dist 
        else:
            return (check3_visibility and check_time)
    
    def get_keyframe_depth(self, cur_frame_idx, depth:torch.Tensor=None, opacity:torch.Tensor=None):
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
    
    def update_keyframe_window(self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window:list):
        near_window_num = 2 # last frame and current frame still in window
        window = [cur_frame_idx] + window
        cur_viewpoint = self.cameras[cur_frame_idx]
        remove_frames_list = []
        removed_frame = None
        # Remove the farthest frame which visibility is low
        for i in range(near_window_num, len(window)): 
            keyframe_idx = window[i]
            intersection = torch.logical_and(cur_frame_visibility_filter, occ_aware_visibility[keyframe_idx]).count_nonzero()
            denom = min(cur_frame_visibility_filter.count_nonzero(), occ_aware_visibility[keyframe_idx].count_nonzero())
            point_ratio = intersection / denom
            cut_off = 0.4 if not self.initialized else self.cut_off
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
       
    def run(self):
        cur_frame_idx = 0
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        
        # NOTE: init feature extractor
        self.set_feature_extractor()
        
        while True:
            # Track and Map pause
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
            if self.pause:
                self.backend_queue.put(["pause"])
                continue
            else:
                self.backend_queue.put(["unpause"])
       
            if self.frontend_queue.empty():
                tic.record()
                # When tracking over
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(self.cameras, self.keyframe_indices, self.save_dir, 0, final=True, monocular=self.monocular)
                        save_gaussians(self.gaussians, self.save_dir, "final", final=True)
                    break
                
                # Check if the mapping is still running
                wait_init_mapping = self.requested_init
                wait_keyframes_mapping = self.single_thread and self.requested_keyframe > 0
                wait_better_mapping = not self.initialized and self.requested_keyframe > 0
                if wait_init_mapping or wait_keyframes_mapping or wait_better_mapping:
                    time.sleep(0.01)
                    continue
                
                # get next frame
                viewpoint = Camera.init_from_dataset(self.dataset, cur_frame_idx, self.projection_matrix)
                viewpoint.compute_grad_mask(self.config)
                self.cameras[cur_frame_idx] = viewpoint
                
                # reset 
                if self.reset:
                    self.initialize_track(cur_frame_idx, viewpoint)
                    
                    cur_frame_idx += 1
                    continue
                
                self.initialized = self.initialized or (len(self.current_window) == self.window_size)

                # tracking for all frame
                start_time = time.time()
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                print(f"[SLAM--Tracking] idx:{cur_frame_idx} time: {time.time() - start_time}")
                
                # after tracking, update the GUI
                self.update_gui_visualize(cur_frame_idx, viewpoint)
                
                # when mapping  TODO
                if self.requested_keyframe > 0: 
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue
                
                last_keyframe_idx = self.current_window[0]
                cur_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(cur_frame_idx, last_keyframe_idx, cur_visibility, self.occ_aware_visibility)
                
                if create_kf:
                    self.current_window, removed = self.update_keyframe_window(cur_frame_idx, 
                                                                               cur_visibility, 
                                                                               self.occ_aware_visibility, 
                                                                               self.current_window)
                
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log("Keyframes lacks sufficient overlap to initialize the map, resetting.")
                        continue
                
                    depth_map = self.get_keyframe_depth(cur_frame_idx, depth=render_pkg["depth"], opacity=render_pkg["opacity"])
                    self.request_backend("keyframe", cur_frame_idx, viewpoint, self.current_window, depth_map)
                else:
                    self.cleanup(cur_frame_idx)
                
                cur_frame_idx += 1
                
                if self.save_results and self.save_trj and create_kf and len(self.keyframe_indices) % self.save_trj_kf_intv == 0:
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.keyframe_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            
            # get message from backend
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)
                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1
                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False
                    self.q_main2vis.put(gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians)))
                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
    
    def request_backend(self, tag:str, cur_frame_idx, viewpoint:Camera, current_window, depth_map):
        if self.semantic_flag:
            # NOTE: [add feature map to the gaussians] cur_semantic_feature: numpy array of shape (H, W, C)
            gt_img = viewpoint.original_image.cuda()
            cur_semantic_feature, cur_vis_feature, _ = self.feature_extractor(gt_img)
            if self.save_semantic: # NOTE： save semantic feature
                semantic_path = os.path.join(self.save_semantic_dir, f"semantic_feature_{cur_frame_idx}.npy")
                np.save(semantic_path, cur_semantic_feature)
                viewpoint.semantic_feature = semantic_path
            else:
                viewpoint.semantic_feature = cur_semantic_feature
            viewpoint.vis_semantic_feature = cur_vis_feature
        msg = [tag, cur_frame_idx, viewpoint, current_window, depth_map]
        self.backend_queue.put(msg)
        if tag == "keyframe":
            self.requested_keyframe += 1
        elif tag == "init":
            self.requested_init = True
        
    def sync_backend(self, data):
        self.gaussians = data[1]
        self.occ_aware_visibility = data[2]
        keyframes = data[3]
        # The mapping process also optimizes the pose of the keyframe
        for kf_id, kf_R, kf_T in keyframes: 
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())
        if self.semantic_flag:
            state_dict_cpu = self.gaussians.semantic_decoder
            self.cnn_decoder.load_state_dict({key: value.cuda() for key, value in state_dict_cpu.items()})
            
    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean() # delete the memory of current frame
        torch.cuda.empty_cache()
    