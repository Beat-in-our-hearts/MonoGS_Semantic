import json
import os
import time
from typing import Dict, List, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F

from PIL import Image
import wandb

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, eval_rendering, eval_segmentation
from utils.logging_utils import Log, debug
from utils.camera_utils import Camera
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth

from utils.semantic_utils import build_decoder
from utils.semantic_setting import Semantic_Config
from utils.semantic_utils import apply_pca_colormap
from imgviz import label_colormap

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue:mp.Queue = None
        self.backend_queue:mp.Queue = None
        self.q_main2vis:mp.Queue = None
        self.q_vis2main:mp.Queue = None
        self.dataset = None
        
        self.semantic_init = False
        self.initialized = False
        self.use_gui = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians:GaussianModel = None
        self.cameras:Dict[int, Camera] = dict()
        self.device = "cuda:0"
        self.pause = False
        
        # CNN Decoder to upsample semantic features
        if Semantic_Config.mode == "SAM2":
            self.cnn_decoder, _ = build_decoder(mode='eval')

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Dataset"]["single_thread"]
        self.eval_rendering = self.config["Results"]["eval_rendering"]
        self.depth_scale = self.config["Dataset"]["Calibration"]["depth_scale"]
        
        self.kf_translation = self.config["Training"]["kf_translation"]
        self.kf_min_translation = self.config["Training"]["kf_min_translation"]
        self.kf_overlap = self.config["Training"]["kf_overlap"]
    
    # def set_feature_extractor(self):
    #     self.feature_extractor = LSeg_FeatureExtractor(debug=True)
    #     self.feature_extractor.eval()
        
    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint:Camera):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        
        # NOTE: init feature extractor
        # self.set_feature_extractor()
        
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False
        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                current_frame=viewpoint,
                gtcolor=viewpoint.original_image.permute(1, 2, 0).cpu().numpy(),
                gtdepth=viewpoint.depth
                if not self.monocular
                else np.zeros((viewpoint.image_height, viewpoint.image_width)),
            ))

    def tracking(self, cur_frame_idx, viewpoint:Camera):
        if self.initialized and cur_frame_idx > Semantic_Config.constant_velocity_warmup:
            prev_prev = self.cameras[cur_frame_idx - self.use_every_n_frames -1 ]
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        
            pose_prev_prev = prev_prev.get_T_matrix4x4
            pose_prev = prev.get_T_matrix4x4
            velocity = pose_prev @ torch.linalg.inv(pose_prev_prev)
            pose_new = velocity @ pose_prev
            viewpoint.update_RT(pose_new[:3, :3], pose_new[:3, 3])
        else:
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            viewpoint.T = prev.T
            
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)

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
        # vis_feature = self.vis_current_frame(viewpoint, cur_frame_idx)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if converged:
                break
        debug(f"Track Iteration: {tracking_itr}")
        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg
    
        
    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
        kf_overlap=0.9,
        only_iou=False,
    ):
        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        # check_time
        check_full_window = len(self.current_window) >= self.window_size
        check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
        # count the distance
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        # Distance check
        dist_check = dist > self.kf_translation * self.median_depth
        dist_check2 = dist > self.kf_min_translation * self.median_depth
        # Common visibility of Gauss points: kf_overlap
        union = torch.logical_or(cur_frame_visibility_filter, 
                                 occ_aware_visibility[last_keyframe_idx]).count_nonzero()
        intersection = torch.logical_and(cur_frame_visibility_filter, 
                                occ_aware_visibility[last_keyframe_idx]).count_nonzero()
        point_ratio_2 = intersection / union
        if only_iou:
            return point_ratio_2 < kf_overlap
        elif check_full_window:
            return (point_ratio_2 < kf_overlap and dist_check2) or dist_check
        else:
            return ((point_ratio_2 < kf_overlap and dist_check2) or dist_check) and check_time

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]

            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
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
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        # TODO
        self.gaussians.load_state_dict(data[1])
        occ_aware_visibility = data[2]
        keyframes = data[3]
        received_state_dict = data[4]
        
        if received_state_dict is not None and self.cnn_decoder is not None:
            self.cnn_decoder.load_state_dict({key: value.cuda() for key, value in received_state_dict.items()})
            self.semantic_init = True
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()
            
    def update_gui(self, viewpoint:Camera):
        if self.use_gui:
            current_window_dict = {}
            current_window_dict[self.current_window[0]] = self.current_window[1:]
            keyframes = [Camera.copy_camera(self.cameras[kf_idx]) for kf_idx in self.current_window]

            decoder_state_dict_cpu = None
            if Semantic_Config.enable and self.semantic_init:
                decoder_state_dict = self.cnn_decoder.state_dict()
                decoder_state_dict_cpu = {key: value.cpu() for key, value in decoder_state_dict.items()}
            
            self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=self.gaussians,
                        gtcolor=viewpoint.original_image.permute(1, 2, 0).cpu().numpy(),
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                        decoder_ckpts=decoder_state_dict_cpu
                    )
                )
    
    def save_render(self, cur_frame_idx, viewpoint:Camera):
        if not self.eval_rendering:
            return 
        
        render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background,
                            flag_semantic=Semantic_Config.enable) 
        
        render_depth = render_pkg["depth"][0]
        rgb_root_dir = os.path.join(self.save_dir, "render", 'rgb')
        depth_root_dir = os.path.join(self.save_dir, "render", 'depth')
        semantic_root_dir = os.path.join(self.save_dir, "render", 'semantic')
        os.makedirs(rgb_root_dir, exist_ok=True)
        os.makedirs(depth_root_dir, exist_ok=True)
        if Semantic_Config.enable:
            os.makedirs(semantic_root_dir, exist_ok=True)
        
        # cv2 save image 
        render_rgb = (
                (torch.clamp(render_pkg["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
        
        render_rgb_path = os.path.join(rgb_root_dir, f"rgb_{cur_frame_idx:04d}.png")
        cv2.imwrite(render_rgb_path, cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR))
        
        render_depth = (render_depth * self.depth_scale).cpu().detach().numpy().astype(np.uint16)
        render_depth_path = os.path.join(depth_root_dir, f"depth_{cur_frame_idx:04d}.png")
        cv2.imwrite(render_depth_path, render_depth)
        

        # TODO
        if Semantic_Config.enable:
            if Semantic_Config.mode == "SAM2":
                feature_map = render_pkg["feature_map"]
                render_shape = feature_map.shape
                resize_feature_map = self.cnn_decoder(F.interpolate(feature_map.unsqueeze(0), 
                                                    size= Semantic_Config.render_size,
                                                    mode="bilinear", align_corners=True).squeeze(0))
                
                sam2_pca = apply_pca_colormap(resize_feature_map.permute(1, 2, 0)).detach().cpu().numpy() # H W C
                img_sam2_pca = (sam2_pca*255).astype(np.uint8)
                img_sam2_pca = cv2.resize(img_sam2_pca, (render_shape[2], render_shape[1]))
                render_semantic_path = os.path.join(semantic_root_dir, f"vis_semantic_{cur_frame_idx:04d}.png")
                cv2.imwrite(render_semantic_path, img_sam2_pca)
            elif Semantic_Config.mode == "GT_Label":
                semantic_class_root_dir = os.path.join(self.save_dir, "render", 'semantic_class')
                os.makedirs(semantic_class_root_dir, exist_ok=True)
                
                feature_map = render_pkg["feature_map"]
                pred_label = torch.argmax(feature_map, dim=0).detach().cpu().numpy()
                img_label = label_colormap()[pred_label]
                
                render_semantic_path = os.path.join(semantic_root_dir, f"vis_semantic_{cur_frame_idx:04d}.png")
                cv2.imwrite(render_semantic_path, cv2.cvtColor(img_label, cv2.COLOR_RGB2BGR))
                semantic_class_path = os.path.join(semantic_class_root_dir, f"semantic_class_{cur_frame_idx:04d}.png")
                cv2.imwrite(semantic_class_path, pred_label.astype(np.uint8))
            else:
                raise NotImplementedError
        debug(f"Saved render: {cur_frame_idx}")
    
    def save_state_dict(self, text):
        if not self.save_results:
            return
        ckpts_dir = os.path.join(self.save_dir, 'ckpts')
        os.makedirs(ckpts_dir, exist_ok=True)
        self.gaussians.save_ply(path=os.path.join(ckpts_dir, f"gaussian_kf_{text}.ply"))
        
        if Semantic_Config.enable:
            if Semantic_Config.mode == "SAM2":
                decoder_state_dict = self.cnn_decoder.state_dict()
                torch.save(decoder_state_dict, os.path.join(ckpts_dir,  f"decoder_{text}.pth"))
        
        pose_dict = {}
        for idx, viewpoint in self.cameras.items():
            pose = {"R": viewpoint.R.cpu().numpy().tolist(), "T": viewpoint.T.cpu().numpy().tolist()}
            pose_dict[idx] = pose
        
        resume_info = {"current_window": self.current_window, "keyframe_indices": self.kf_indices, "pose_dict": pose_dict}
        with open(os.path.join(ckpts_dir, f"resume_info_{text}.json"), 'w', encoding='utf-8') as f:
            json.dump(resume_info, f, indent=4)
            

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
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
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results: 
                        self.save_state_dict("final")
                    break

                if self.requested_init:
                    time.sleep(0.5)
                    debug('waiting for init')
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.1)
                    debug('sp: waiting for keyframe')
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.1)
                    debug('init: waiting for keyframe')
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += self.use_every_n_frames
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                track_start_time = time.time()
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                debug(f"[{cur_frame_idx:04d}] track time: {time.time()-track_start_time}")

                self.save_render(cur_frame_idx, viewpoint)
                
                # update GUI
                self.update_gui(viewpoint)

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += self.use_every_n_frames
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                    kf_overlap=self.kf_overlap,
                    only_iou=Semantic_Config.kf_only_iou
                )
                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
    
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += self.use_every_n_frames

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    self.save_state_dict(f"{self.kf_indices[-1]:04d}")
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    all_frame_id = list(range(self.kf_indices[-1]))
                    ate_result = eval_ate(
                        self.cameras,
                        all_frame_id,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                    rendering_result = eval_rendering(
                        self.cameras,
                        self.gaussians,
                        self.dataset,
                        self.save_dir,
                        self.pipeline_params,
                        self.background,
                        kf_indices=self.kf_indices,
                        iteration="before_opt",
                        depth_l1=not self.monocular
                    )
                    if Semantic_Config.eval_segmentation:
                        seg_result = eval_segmentation(
                            self.cameras,
                            self.dataset,
                            self.gaussians,
                            self.pipeline_params,
                            self.background,
                        )
                    else:
                        seg_result = {"pixel_acc": 0, "mIoU": 0}
                    kf_idx = self.kf_indices[-1]
                    kf_output = {
                        "frame_idx": kf_idx,
                        "rmse_ate": ate_result["rmse"],
                        "mean_ate": ate_result["mean"],
                        "psnr": rendering_result["mean_psnr"],
                        "ssim": rendering_result["mean_ssim"],
                        "lpips": rendering_result["mean_lpips"],
                        "depth_l1": rendering_result["mean_depth_l1"],
                        "seg_pix_acc": seg_result["pixel_acc"],
                        "seg_mIoU": seg_result["mIoU"],
                    }
                    wandb.log(kf_output)
                    if self.save_results:
                        metric_dir = os.path.join(self.save_dir, "metric")
                        os.makedirs(metric_dir, exist_ok=True)
                        with open(os.path.join(metric_dir, f"eval_kf_{kf_idx:04d}.json"), 'w', encoding='utf-8') as f:
                            json.dump(kf_output, f, indent=4)          
                    
                toc.record()
                if Semantic_Config.synchronize:
                    torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    sleep_time = 1.0 / 3.0 - duration / 1000
                    time.sleep(max(0.01, sleep_time))
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
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=self.gaussians,
                        )
                    )
                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
