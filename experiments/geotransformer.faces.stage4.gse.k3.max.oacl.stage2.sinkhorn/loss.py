import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import os
from pytorch3d.loss import chamfer_distance
from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.modules.ops.transformation import apply_transform
from geotransformer.modules.registration.metrics import isotropic_transform_error
from geotransformer.modules.ops.pairwise_distance import pairwise_distance

class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss
    
    
class MorphableLoss(nn.Module):
    def __init__(self, cfg):
        super(MorphableLoss, self).__init__()
        self.mae_loss = nn.L1Loss(reduction='sum')

    def forward(self, output_dict, data_dict, epoch=None, iteration=None, mode='train'):
        # Get prediction (morphed ref) and ground truth (src full pcd)
        pred_points = output_dict['morphed_full'] 
        gt_points = data_dict['morphed_full']

        pred_z = output_dict['z_coefficients']
        gt_z = data_dict['gt_z']
        
        # Format for Pytorch3D (batch, num_points, dim)
        if pred_points.dim() == 2:
            pred_points = pred_points.unsqueeze(0)
        if gt_points.dim() == 2:
            gt_points = gt_points.unsqueeze(0)

        """
        # Visualization block
        recon_gt_points = output_dict['recon_gt_points']

        do_viz = False
        if iteration is not None:
            if mode == 'train' and iteration in [10, 20, 30, 40, 50, 1010, 1020, 1030, 1040, 1050]:
                do_viz = True
            elif mode == 'val' and iteration in [5, 15, 25, 35, 45]:
                do_viz = True

        if do_viz:
            import os
            import numpy as np
            import matplotlib
            matplotlib.set_loglevel('warning')
            import matplotlib.pyplot as plt

            from datetime import datetime

            # Save directory
            viz_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'viz_debug')
            os.makedirs(viz_dir, exist_ok=True)

            file_name_base = f"viz_{mode}_epoch_{epoch:04d}" if epoch is not None else f"viz_{mode}_debug"

            # Convert to numpy (Assuming shape [N, 3] based on our previous discussion)
            pred_np = pred_points.detach().cpu().numpy().squeeze()
            recon_gt_np = recon_gt_points.detach().cpu().numpy().squeeze()
            gt_np = gt_points.detach().cpu().numpy().squeeze()

            if pred_np.ndim == 1:
                pred_np = pred_np[np.newaxis, :]

            views = [("front", 0, 0)]

            for name, elev, azim in views:
                # Widen the figure to accommodate 3 plots
                fig = plt.figure(figsize=(18, 6))

                # Plot 1: Prediction (from pred_z)
                ax1 = fig.add_subplot(131, projection='3d')
                ax1.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], c='r', s=1)
                ax1.set_title(f"Prediction (pred_z) ({name})")
                ax1.view_init(elev=elev, azim=azim)

                # Plot 2: Reconstruction from GT Z (New)
                ax2 = fig.add_subplot(132, projection='3d')
                ax2.scatter(recon_gt_np[:, 0], recon_gt_np[:, 1], recon_gt_np[:, 2], c='g', s=1)
                ax2.set_title(f"Reconstruction (gt_z) ({name})")
                ax2.view_init(elev=elev, azim=azim)

                # Plot 3: Original Ground Truth Point Cloud
                ax3 = fig.add_subplot(133, projection='3d')
                ax3.scatter(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2], c='b', s=1)
                ax3.set_title(f"Ground Truth Point Cloud ({name})")
                ax3.view_init(elev=elev, azim=azim)

                timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                candidate = os.path.join(viz_dir, f"{file_name_base}_it{iteration:06d}_{timestamp}_{name}.png")

                suffix = 0
                save_path = candidate
                while os.path.exists(save_path):
                    suffix += 1
                    save_path = candidate.replace(".png", f"_{suffix:02d}.png")

                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"Saved visualization: {save_path}")
        # End visualization block
        """

        # Chamfer Distance
        #loss_chamfer, _ = chamfer_distance(pred_points, gt_points)

        # MSE
        #loss_mae = self.mae_loss(pred_points, gt_points)

        MSE = F.mse_loss(pred_z, gt_z)

        return MSE
    



class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.morph_loss = MorphableLoss(cfg)

        self.weight_coarse_loss = 1.0 #cfg.loss.weight_coarse_loss
        self.weight_fine_loss = 1.0 #cfg.loss.weight_fine_loss
        self.weight_morph_loss = 1.0   # update as needed 

    def forward(self, output_dict, data_dict, epoch=None, iteration=None, mode='train'):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)
        morph_loss = self.morph_loss(output_dict, data_dict, epoch, iteration, mode=mode)
        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss + self.weight_morph_loss * morph_loss

        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
            'm_loss': morph_loss
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        src_points = output_dict['src_points']
        
        # ==================================================
        # Normalize the rotation blocks to remove scale
        def normalize_transform(T):
            T_norm = T.clone()
            # Calculate scale as the norm of the first column of the 3x3 block
            s = torch.norm(T[:3, 0]) 
            T_norm[:3, :3] = T[:3, :3] / s
            return T_norm, s
        
        gt_norm, gt_scale = normalize_transform(transform)
        est_norm, est_scale = normalize_transform(est_transform)
        rre, rte = isotropic_transform_error(gt_norm, est_norm)

        #===================================================

        #rre, rte = isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)

        return {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
        }
