import json
import os
import time

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim, l1_loss
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log

from utils.semantic_setting import Semantic_Config
from utils.eval_segmentation import SegmentationMetric
from diff_gaussian_rasterization import get_semantic_channels
from utils.semantic_utils import build_decoder

def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ate_mean = ape_metric.get_statistic(metrics.StatisticsType.mean)
    ape_stats = ape_metric.get_all_statistics()
    
    Log(
        f'RMSE ATE \[cm]: {(ape_stat*100):.3f}, ' + f'Mean ATE \[cm]: {(ate_mean*100):.3f}',
        tag="Eval",
    )

    # with open(
    #     os.path.join(plot_dir, "stats_{}.json".format(str(label))),
    #     "w",
    #     encoding="utf-8",
    # ) as f:
    #     json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.close(fig)
    return {'rmse': ape_stat, 'mean': ate_mean}


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "ate_plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    # with open(
    #     os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    # ) as f:
    #     json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    # wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate

@torch.no_grad()
def benchmark_render_time(frame, gaussians, pipe, background, num_iter=2000, flag_semantic=False):
    start_time = time.time()
    for _ in range(num_iter):
        render(frame, gaussians, pipe, background, flag_semantic=flag_semantic)
    end_time = time.time()
    FPS = num_iter / (end_time - start_time)
    Log(f"Render FPS: {FPS:.1f}", tag="Eval")
    return FPS


@torch.no_grad()
def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
    depth_l1=False,
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array, depth_l1_array = [], [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    for idx in range(0, end_idx, interval):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, gt_depth, _ = dataset[idx]

        render_pkg = render(frame, gaussians, pipe, background)
        rendering = render_pkg["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

        if depth_l1:
            gt_depth = torch.tensor(gt_depth).cuda()
            depth_pixel_mask = (gt_depth > 0.01).view(*gt_depth.shape)
            opacity_mask = (render_pkg["opacity"] > 0.95).view(*gt_depth.shape)
            depth_mask = depth_pixel_mask * opacity_mask
            render_depth = render_pkg["depth"][0]
            depth_l1_score = l1_loss(render_depth[depth_mask], gt_depth[depth_mask])
            depth_l1_array.append(depth_l1_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["mean_depth_l1"] = float(np.mean(depth_l1_array)) if depth_l1 else 0

    Log(
        f'mean psnr: {output["mean_psnr"]:.2f}, ' + f'ssim: {output["mean_ssim"]:.3f}, ' + \
        f'lpips: {output["mean_lpips"]:.4f}, ' + f'depth_l1: {output["mean_depth_l1"]*100:.3f}',
        tag="Eval",
    )
    return output



def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(point_cloud_path + "_point_cloud.ply")

def eval_segmentation(frames, dataset, gaussians, pipe, background, save_dir=None, decoder_state_dict=None, clip_text_feature=None):
    seg_metric = SegmentationMetric(nclass=get_semantic_channels())
    if save_dir is not None:
        semantic_class_root_dir = os.path.join(save_dir, "render", 'eval_semantic_class')
        os.makedirs(semantic_class_root_dir, exist_ok=True)
    
    if Semantic_Config.mode == "Grounding_Dino":
        cnn_decoder, _ = build_decoder(mode="eval")
        if decoder_state_dict is not None:
            cnn_decoder.load_state_dict(decoder_state_dict)
        else:
            raise ValueError("Decoder state dict is None.")
        if clip_text_feature is None:
            raise ValueError("Clip text feature is None.")
        
    for idx in range(len(frames)):
        frame = frames[idx]
        
        if Semantic_Config.mode == "GT_Label":
            render_pkg = render(frame, gaussians, pipe, background, flag_semantic=True)
            feature_map = render_pkg["feature_map"]
            pred_label = torch.argmax(feature_map, dim=0).detach().cpu().numpy().astype(np.uint8)
            
            gt_label_path = dataset.get_gt_semantic(idx)
            gt_label = cv2.imread(gt_label_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            seg_metric.update(pred_label, gt_label)
            
            if save_dir is not None:    
                semantic_class_path = os.path.join(semantic_class_root_dir, f"semantic_class_{idx:04d}.png")
                cv2.imwrite(semantic_class_path, pred_label.astype(np.uint8))
                
        elif Semantic_Config.mode == "Grounding_Dino":
            render_pkg = render(frame, gaussians, pipe, background, flag_semantic=True)
            feature_map = render_pkg["feature_map"]
            feature_map = cnn_decoder(feature_map)
            pred_ssim = feature_map.permute(1, 2, 0) @ clip_text_feature.T
            threshold = 0.6
            black_mask = (pred_ssim < threshold).all(dim=-1)
            pred_label = (torch.argmax(pred_ssim, dim=-1) + 1) # W x H, 0 is background
            pred_label[black_mask] = 0 # 0 is background
            pred_label = pred_label.detach().cpu().numpy().astype(np.uint8)
            
            gt_label_path = dataset.get_gt_semantic(idx)
            gt_label = cv2.imread(gt_label_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            
            # TODO fix
            synonyms_id_dict = {98:40, 97:12}
            for key, value in synonyms_id_dict.items():
                pred_label[pred_label == key] = value
                gt_label[gt_label == key] = value
            
            seg_metric.update(pred_label, gt_label)
            if save_dir is not None:    
                semantic_class_path = os.path.join(semantic_class_root_dir, f"pred_semantic_class_{idx:04d}.png")
                cv2.imwrite(semantic_class_path, pred_label.astype(np.uint8))
            
    pixel_acc, mIoU = seg_metric.get()
    Log(
        f'pixel_acc: {pixel_acc:.3f}, ' + f'mIoU: {mIoU:.3f}',
        tag="Eval",
    )
    return {'pixel_acc': pixel_acc, 'mIoU': mIoU}