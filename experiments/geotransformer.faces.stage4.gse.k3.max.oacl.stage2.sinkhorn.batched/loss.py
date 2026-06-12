import torch
import torch.nn as nn
import torch.nn.functional as F

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
        ref_feats_b = output_dict['ref_feats_c']      # (B, N_ref_c, C)
        src_feats_list = output_dict['src_feats_c']   # list of (n_i, C)
        batch_size = len(src_feats_list)

        losses = []
        for i in range(batch_size):
            gt_node_corr_indices = output_dict['gt_node_corr_indices'][i]
            gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps'][i]
            gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
            gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

            feat_dists = torch.sqrt(pairwise_distance(ref_feats_b[i], src_feats_list[i], normalized=True))

            overlaps = torch.zeros_like(feat_dists)
            overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
            pos_masks = torch.gt(overlaps, self.positive_overlap)

            if pos_masks.sum() == 0:
                continue

            neg_masks = torch.eq(overlaps, 0)
            pos_scales = torch.sqrt(overlaps * pos_masks.float())

            losses.append(self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales))

        if len(losses) == 0:
            return torch.tensor(0.0, device=ref_feats_b.device, requires_grad=True)

        return torch.stack(losses).mean()


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
        if transform.dim() == 2:
            transform = transform.unsqueeze(0)
        # Per-item transform expanded to one (4, 4) per node correspondence row.
        counts = output_dict['node_corr_counts'].to(transform.device)
        row_transforms = torch.repeat_interleave(transform, counts, dim=0)  # (sum_P, 4, 4)

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, row_transforms)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (sum_P, K, K)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))

        with torch.no_grad():
            gt_corr_map = torch.lt(dists, self.positive_radius ** 2) & gt_masks
            slack_row_labels = torch.logical_and(gt_corr_map.sum(2).eq(0), ref_node_corr_knn_masks)
            slack_col_labels = torch.logical_and(gt_corr_map.sum(1).eq(0), src_node_corr_knn_masks)
            hard_pos_mask = gt_corr_map.float()

        pos_loss = -(matching_scores[:, :-1, :-1] * hard_pos_mask).sum() / hard_pos_mask.sum().clamp(min=1e-6)

        # Hard slack (dustbin) loss — unchanged behavior
        slack_scores = []
        if slack_row_labels.any():
            slack_scores.append(matching_scores[:, :-1, -1][slack_row_labels])
        if slack_col_labels.any():
            slack_scores.append(matching_scores[:, -1, :-1][slack_col_labels])
        slack_loss = (-torch.cat(slack_scores).mean()
                      if slack_scores else torch.tensor(0.0, device=matching_scores.device))

        return pos_loss + slack_loss


class MorphableLoss(nn.Module):
    def __init__(self, cfg):
        super(MorphableLoss, self).__init__()
        self.num_pca_components = cfg.model.num_pca_components

    def forward(self, output_dict, data_dict, epoch=None, iteration=None, mode='train'):
        pred_z = output_dict['z_coefficients']                            # (B, 32, n_comp)
        gt_z = data_dict['gt_z']
        if gt_z.dim() == 2:
            gt_z = gt_z.unsqueeze(0)
        gt_z = gt_z[:, :, :self.num_pca_components]                       # (B, 32, n_comp)

        loss_z = F.mse_loss(pred_z, gt_z)

        # Dense vertex-to-vertex loss: predicted morphed mesh vs GT morphed mesh.
        pred_morphed = output_dict['morphed_full_grad']                   # (B, V, 3)
        gt_morphed = data_dict['morphed_full']
        if gt_morphed.dim() == 2:
            gt_morphed = gt_morphed.unsqueeze(0)
        loss_dense = F.l1_loss(pred_morphed, gt_morphed.to(pred_morphed.device))

        return loss_z, loss_dense


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.morph_loss = MorphableLoss(cfg)

        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss
        self.weight_morph_loss = cfg.loss.weight_morph_loss
        self.weight_dense_loss = cfg.loss.weight_dense_loss

    def forward(self, output_dict, data_dict, epoch=None, iteration=None, mode='train'):
        morph_loss, dense_loss = self.morph_loss(output_dict, data_dict, epoch, iteration, mode=mode)

        coarse_loss = torch.tensor(0.0).cuda()
        fine_loss = torch.tensor(0.0).cuda()

        if mode == 'train' and epoch is not None:
            if epoch > 1 or (epoch == 1 and iteration is not None and iteration >= 250):
                coarse_loss = self.coarse_loss(output_dict)
                fine_loss = self.fine_loss(output_dict, data_dict)
        else:
            coarse_loss = self.coarse_loss(output_dict)
            fine_loss = self.fine_loss(output_dict, data_dict)

        loss = (self.weight_coarse_loss * coarse_loss +
                self.weight_fine_loss * fine_loss +
                self.weight_morph_loss * morph_loss +
                self.weight_dense_loss * dense_loss)

        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
            'm_loss': morph_loss,
            'd_loss': dense_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[1]
        batch_size = len(output_dict['src_points_c'])

        precisions = []
        for i in range(batch_size):
            src_length_c = output_dict['src_points_c'][i].shape[0]
            gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps'][i]
            gt_node_corr_indices = output_dict['gt_node_corr_indices'][i]
            masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
            gt_node_corr_indices = gt_node_corr_indices[masks]
            gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
            gt_node_corr_map[gt_node_corr_indices[:, 0], gt_node_corr_indices[:, 1]] = 1.0

            ref_node_corr_indices = output_dict['ref_node_corr_indices'][i]
            src_node_corr_indices = output_dict['src_node_corr_indices'][i]

            precisions.append(gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean())

        return torch.stack(precisions).mean()

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        if transform.dim() == 2:
            transform = transform.unsqueeze(0)
        batch_size = transform.shape[0]

        all_distances = []
        for i in range(batch_size):
            ref_corr_points = output_dict['ref_corr_points'][i]
            src_corr_points = output_dict['src_corr_points'][i]
            if ref_corr_points.shape[0] == 0:
                continue
            src_corr_points = apply_transform(src_corr_points, transform[i])
            all_distances.append(torch.linalg.norm(ref_corr_points - src_corr_points, dim=1))

        if len(all_distances) == 0:
            return torch.tensor(0.0).cuda()
        corr_distances = torch.cat(all_distances, dim=0)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        if transform.dim() == 2:
            transform = transform.unsqueeze(0)
        est_transforms = output_dict['estimated_transform']               # (B, 4, 4)
        batch_size = transform.shape[0]

        rre_list, rte_list, rmse_list, recall_list = [], [], [], []
        for i in range(batch_size):
            rre, rte = isotropic_transform_error(transform[i], est_transforms[i])

            realignment_transform = torch.matmul(torch.inverse(transform[i]), est_transforms[i])
            src_points = output_dict['src_points'][i]
            realigned_src_points = apply_transform(src_points, realignment_transform)
            rmse = torch.linalg.norm(realigned_src_points - src_points, dim=1).mean()

            rre_list.append(rre)
            rte_list.append(rte)
            rmse_list.append(rmse)
            recall_list.append(torch.lt(rmse, self.acceptance_rmse).float())

        return (
            torch.stack(rre_list).mean(),
            torch.stack(rte_list).mean(),
            torch.stack(rmse_list).mean(),
            torch.stack(recall_list).mean(),
        )

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
