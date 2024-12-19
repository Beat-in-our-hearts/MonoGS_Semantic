import argparse
import json
import os
from tqdm import tqdm
from typing import Dict, List, Union
from PIL import Image
import cv2
import evo
import numpy as np
import torch
import torch.nn as nn
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim, l1_loss
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log
from utils.camera_utils import Camera

from utils.common_var import *
from eval.segmentationMetric import SegmentationMetric
from feature_encoder.lseg_encoder.feature_extractor import LSeg_FeatureDecoder

def gen_pose_matrix(R, T):
    pose = np.eye(4)
    pose[0:3, 0:3] = R.cpu().numpy()
    pose[0:3, 3] = T.cpu().numpy()
    return pose 

parent_directory = os.path.dirname(os.path.abspath(__file__))
labels_path = os.path.join(parent_directory, 'labels/replica_ade20k_match_dict.json')

def Eval_Tracking(cameras:Dict[int, Camera], save_dir=None, monocular=False):
    traj_est: List[np.ndarray] = []
    traj_gt: List[np.ndarray] = []
    
    # convert to numpy array
    traj_est = [np.linalg.inv(gen_pose_matrix(frame.R, frame.T)) for frame in cameras.values()]
    traj_gt  = [np.linalg.inv(gen_pose_matrix(frame.R_gt, frame.T_gt)) for frame in cameras.values()] 
    
    # convert to PosePath3D
    pose_gt = PosePath3D(poses_se3=traj_gt)
    pose_est = PosePath3D(poses_se3=traj_est)
    pose_est_aligned = trajectory.align_trajectory(pose_est, pose_gt, correct_scale=monocular)
    
    # define the metric
    pose_relation = metrics.PoseRelation.translation_part
    ate_metric = metrics.APE(pose_relation)
    
    # process the data
    pose_data = (pose_gt, pose_est_aligned)
    ate_metric.process_data(pose_data)
    ate_rmse = ate_metric.get_statistic(metrics.StatisticsType.rmse)
    ate_mean = ate_metric.get_statistic(metrics.StatisticsType.mean)
    ate_stats = ate_metric.get_all_statistics()
    
    if save_dir is not None:
        Log(f"ATE RMSE [cm]: {ate_rmse*100}", tag="Eval")
        Log(f"ATE Mean [cm]: {ate_mean*100}", tag="Eval")
        
        # Write the trajectory data to a json file
        with open(os.path.join(save_dir, 'ATE_traj.json'), 'w', encoding='utf-8') as f:
            traj_gt_data = [traj.tolist() for traj in traj_gt]
            traj_est_data = [traj.tolist() for traj in traj_est]
            traj_data = {"traj_gt": traj_gt_data, "traj_est": traj_est_data}
            json.dump(traj_data, f, indent=4)
            
        # Write the results to a json file
        with open(os.path.join(save_dir, 'ATE_result.json'), 'w', encoding='utf-8') as f:
            json.dump(ate_stats, f, indent=4)
            
        # Plot the results
        plot_mode = evo.tools.plot.PlotMode.xy
        fig = plt.figure()
        ax = evo.tools.plot.prepare_axis(fig, plot_mode)
        ax.set_title(f"ATE RMSE: {ate_rmse*100} cm")
        evo.tools.plot.traj(ax, plot_mode, pose_gt, "--", "gray", "gt")
        evo.tools.plot.traj_colormap(ax, pose_est_aligned, ate_metric.error, plot_mode, min_map=ate_stats["min"], max_map=ate_stats["max"])
        ax.legend()
        plt.savefig(os.path.join(save_dir, "ATE_plot.png"), dpi=90)
        
    return ate_stats

def Eval_Mapping(cameras:Dict[int, Camera], dataset, 
                gaussians, pipeline_params, bg_color,
                save_dir=None, monocular=False, interval=5):
    PSNR, SSIM, LPIPS = [], [], [] # Rendering Quality Metrics
    DEPTH_L1 = [] # Reconstruction Metrics
    
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
    
    for idx, frame in tqdm(cameras.items(), desc="Evaluating Mapping"):

        gt_color, gt_depth, _ = dataset[idx]
        
        gt_shape = gt_color.shape
        mask = gt_color > 0.01

        
        with torch.no_grad():
            render_pkg = render(frame, gaussians, pipeline_params, bg_color)
        
            # Rendering Metrics
            render_color = torch.clamp(render_pkg["render"], 0.0, 1.0)
            render_depth = render_pkg["depth"]
            gt_depth = torch.tensor(gt_depth, device=render_depth.device).unsqueeze(0)
            
            render_color_with_mask = torch.zeros_like(render_color)
            gt_color_with_mask = torch.zeros_like(gt_color)

            render_color_with_mask[mask] = render_color[mask]
            gt_color_with_mask[mask] = gt_color[mask]

            psnr_score = psnr((render_color_with_mask).unsqueeze(0), (gt_color_with_mask).unsqueeze(0))
            ssim_score = ssim((render_color_with_mask).unsqueeze(0), (gt_color_with_mask).unsqueeze(0))
            lpips_score = cal_lpips((render_color_with_mask).unsqueeze(0), (gt_color_with_mask).unsqueeze(0))
            
            PSNR.append(psnr_score.item())
            SSIM.append(ssim_score.item())
            LPIPS.append(lpips_score.item())
            
            # Reconstruction Metrics
            depth_pixel_mask = (gt_depth > 0.01).view(*gt_depth.shape)
            opacity_mask = (render_pkg["opacity"] > 0.95).view(*gt_depth.shape)
            depth_mask = depth_pixel_mask * opacity_mask
            depth_l1 = l1_loss(render_depth[depth_mask], gt_depth[depth_mask])
            
            DEPTH_L1.append(depth_l1.item())

    # Compute the mean of the metrics
    mean_psnr = np.mean(PSNR)
    mean_ssim = np.mean(SSIM)
    mean_lpips = np.mean(LPIPS)
    mean_depth_l1 = np.mean(DEPTH_L1)
    output = {"mean_psnr": mean_psnr, "mean_ssim": mean_ssim, "mean_lpips": mean_lpips, "mean_depth_l1": mean_depth_l1}
    
    with open("map.log", "a+") as f:
        f.write(f"Mean PSNR: {PSNR}\n")
        f.write(f"Mean SSIM: {SSIM}\n")
        f.write(f"Mean LPIPS: {LPIPS}\n")
        f.write(f"Mean Depth L1: {DEPTH_L1}\n\n")
    
    if save_dir is not None:
        # Write the results to a json file
        with open(os.path.join(save_dir, 'rendering_result.json'), 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4)
        
        # Log the results
        Log(f"Depth L1 \[cm]: {mean_depth_l1*100}", tag="Eval")
        Log(f"PNSR \[dB]: {mean_psnr}", tag="Eval")
        Log(f"SSIM: {mean_ssim}", tag="Eval")
        Log(f"LPIPS: {mean_lpips}", tag="Eval")

    torch.cuda.empty_cache()
    return output

def transform_labels(labels, id_dict):
    reslut = labels.copy()
    for key, value in id_dict.items():
        reslut[labels == key] = value
    return reslut

# TODO
def Eval_Semantic(cameras:Dict[int, Camera], dataset, 
                    gaussians, pipeline_params, bg_color, 
                    save_dir=None, monocular=False, interval=5):
    pixAcc_list = []
    mIoU_list = []
    # Load the feature upsample decoder
    cnn_decoder = nn.Conv2d(SEMANTIC_FEATURES_DIM, Distilled_Feature_DIM, kernel_size=1).to("cuda")
    state_dict_cpu = gaussians.semantic_decoder
    cnn_decoder.load_state_dict({key: value.cuda() for key, value in state_dict_cpu.items()})
    cnn_decoder.eval()
    # Load the feature encoder
    feature_decoder = LSeg_FeatureDecoder(debug=False)
    # print(feature_decoder.labels)
    # Load seg_metric
    seg_metric = SegmentationMetric(100)
    # Load label transform 
    with open(labels_path) as f:
        replica_ade20k_match_dict = json.load(f)
    ade20k_to_replica_id = {int(key): int(value) for key, value in replica_ade20k_match_dict["ade20k_to_replica"].items()}
    # replica_id_to_name = {int(key): value for key, value in replica_ade20k_match_dict["objects"].items()}
    replica_synonyms_transform = {int(key): int(value) for key, value in replica_ade20k_match_dict["replica_synonyms_transform"].items()}
    
    with torch.no_grad():
        with tqdm(total=len(cameras), desc="Evaluating Semantic") as pbar:
            for idx, frame in cameras.items():
                render_pkg = render(frame, gaussians, pipeline_params, bg_color, flag_semantic=True)
                feature_map = render_pkg["feature_map"]
                upsample_feature_map = cnn_decoder(feature_map)
                # fmap_path = f"fmap_{idx}.pt"
                # torch.save(upsample_feature_map.cpu(), fmap_path)
                output = feature_decoder.features_to_image(upsample_feature_map)
                predict = output["predict"][0] + 1
                predict_semantic_transform = transform_labels(predict, ade20k_to_replica_id)
                
                # Get GT and Mask
                gt_semantic_path = dataset.get_gt_semantic_path(idx)
                gt_semantic = cv2.imread(gt_semantic_path, cv2.IMREAD_GRAYSCALE)
                gt_semantic_transform = transform_labels(gt_semantic, replica_synonyms_transform)
                valid_mask = gt_semantic_transform > 0
                
                # cv2 save the semantic image
                # print(predict_semantic_transform.shape, gt_semantic_transform.shape)
                # np.save(f"predict_{idx}.npy", predict_semantic_transform)
                # np.save(f"gt_{idx}.npy", gt_semantic_transform)
                
                # Update metrics
                # print(predict_semantic_transform.shape, gt_semantic_transform.shape)
                seg_metric.update(torch.tensor(predict_semantic_transform), torch.tensor(gt_semantic_transform))
                pixAcc, mIoU = seg_metric.get()
                pixAcc_list.append(pixAcc)
                mIoU_list.append(mIoU)
                pbar.update(1)
                pbar.set_description(f"Evaluating Semantic pixAcc: {pixAcc:.4f}, mIoU: {mIoU:.4f}")
                torch.cuda.empty_cache()
    
    pixAcc, mIoU = seg_metric.get()
    output = {"pixAcc": pixAcc, "mIoU": mIoU}
    if save_dir is not None:
        Log(f"pixAcc: {pixAcc} mIoU: {mIoU}", tag="Eval")
        all_output = {"pixAcc": pixAcc_list, "mIoU": mIoU_list}
        with open(os.path.join(save_dir, 'semantic_result.json'), 'w', encoding='utf-8') as f:
            json.dump(all_output, f, indent=4)
    return output

def run_all_metrics(cameras:Dict[int, Camera], dataset, 
                    gaussians, pipeline_params, bg_color,
                    save_dir, monocular=False, interval=5):
    # Evaluate Tracking
    tracking_stats = Eval_Tracking(cameras, save_dir, monocular)
    
    # Evaluate Mapping
    mapping_stats = Eval_Mapping(cameras, dataset, gaussians, pipeline_params, bg_color, save_dir, monocular, interval)
    
    # Evaluate Semantic
    semantic_stats = Eval_Semantic(cameras, dataset, gaussians, pipeline_params, bg_color, save_dir, monocular, interval)
    
    return tracking_stats, mapping_stats, semantic_stats

def main():
    parser = argparse.ArgumentParser("Evaluate the performance of the SLAM system")
    parser.add_argument("--eval_config", type=str, required=True, help="The path to the evaluation configuration file")
    
    args = parser.parse_args()
    raise NotImplementedError("Evaluation is not implemented yet")