import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.utils.data import precompute_data_stack_mode
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)
from pytorch3d.ops import sample_farthest_points
from backbone import KPConvFPN
from coeff_regressor import CrossAttentionRegressor, generate_reference_geometry


class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        pca_data = torch.load("pca_basis_all.pth")

        self.register_buffer("pca_basis", pca_data['basis']) 
        self.register_buffer("pca_mean", pca_data['mean'])  
        self.register_buffer("patch_indices", pca_data['patch_indices']) 

        self.num_patches = self.patch_indices.shape[0]
        self.num_components = self.pca_basis.shape[1]

        gt_z_mean = torch.mean(pca_data['gt_z'], dim=1)
        self.register_buffer("z_template", gt_z_mean)

        self.coeff_encoder_type = cfg.model.coeff_encoder_type
        self.coeff_regressor = CrossAttentionRegressor(
            feature_dim=cfg.model.coeff_regressor_feature_dim,
            num_patches=32,
            num_coeffs=100,
            nhead=4,
            num_layers=2,
            encoder_type=cfg.model.coeff_encoder_type,
            num_sampled_points=cfg.model.coeff_regressor_sampled_points,
            k_neighbors=cfg.model.coeff_regressor_k_neighbors,
            dropout=cfg.model.coeff_regressor_dropout,
        )

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
            use_umeyama=cfg.model.use_umeyama,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

        # Backbone graph hyperparams — used to rebuild the graph in forward()
        self.num_stages = cfg.backbone.num_stages
        self.init_voxel_size = cfg.backbone.init_voxel_size
        self.init_radius = cfg.backbone.init_radius
        self.neighbor_limits = None  # set by trainer after construction
        self.max_superpoints = cfg.model.max_superpoints

    def generate_reference_geometry(self, z_delta):
        return generate_reference_geometry(z_delta, self.pca_basis, self.pca_mean, self.patch_indices)

    @staticmethod
    def _mem(label):
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / 1e6
        peak  = torch.cuda.max_memory_allocated() / 1e6
        # print(f"[MEM] {label:<45s}  alloc={alloc:7.1f} MB  peak={peak:7.1f} MB")

    def forward(self, data_dict):
        output_dict = {}
        torch.cuda.reset_peak_memory_stats()
        self._mem("start")

        # --- 1. Extract raw src ---
        # Robustly handle both precomputed lists and raw flat tensors
        if isinstance(data_dict['lengths'], list):
            # Dataloader is still precomputing! Extract the stage 0 flat tensors.
            flat_lengths = data_dict['lengths'][0]
            flat_points = data_dict['points'][0]
        else:
            # precompute_data=False worked as intended
            flat_lengths = data_dict['lengths']
            flat_points = data_dict['points']

        ref_length_stage0 = flat_lengths[0].item()
        src_points_raw = flat_points[ref_length_stage0:]

        # --- 2. Predict scale & coeffs from raw coords ---
        num_samples = 1024
        num_src_points = src_points_raw.shape[0]

        if num_src_points > num_samples:
            src_coords_batched, _ = sample_farthest_points(src_points_raw.unsqueeze(0), K=num_samples)
        else:
            src_coords_batched = src_points_raw.unsqueeze(0)

        padding_mask = torch.zeros((1, src_coords_batched.shape[1]), dtype=torch.bool, device=src_points_raw.device)
        centroid = src_coords_batched.mean(dim=1, keepdim=True)
        src_coords_centered = src_coords_batched - centroid

        self._mem("before coeff_regressor")
        z_delta_batched, pred_scale_batched = self.coeff_regressor(src_coords_centered, padding_mask)
        self._mem("after  coeff_regressor")

        z_delta = z_delta_batched.squeeze(0)
        pred_scale = pred_scale_batched.squeeze()

        output_dict['z_coefficients'] = z_delta
        output_dict['pred_scale'] = pred_scale

        # --- 3. Morph Ref Points and Dynamically Compute Graph ---
        with torch.no_grad():
            new_ref_points = self.generate_reference_geometry(z_delta.detach())
            output_dict['morphed_full'] = new_ref_points

            gt_z = data_dict['gt_z']
            recon_gt_points = self.generate_reference_geometry(gt_z)
            output_dict['recon_gt_points'] = recon_gt_points

            src_points_scaled = src_points_raw / pred_scale.detach()

            # Combine morphed ref and scaled src into a single tensor
            new_points = torch.cat([new_ref_points, src_points_scaled], dim=0)

            # Extract the device so we can return the graph to the GPU
            device = new_points.device

            # The graph-building ops expect tensors on the CPU
            graph_dict = precompute_data_stack_mode(
                new_points.cpu(),
                flat_lengths.cpu(),
                self.num_stages,
                self.init_voxel_size,
                self.init_radius,
                self.neighbor_limits
            )

            # Move the computed lists of tensors back to the GPU
            for key in ['points', 'lengths', 'neighbors', 'subsampling', 'upsampling']:
                if key in graph_dict:
                    graph_dict[key] = [t.to(device) for t in graph_dict[key]]

            # Overwrite the flat data_dict properties with the newly computed graph lists
            data_dict.update(graph_dict)
            self._mem("after  graph build + GPU transfer")

        # --- 4. Run Backbone on dynamically computed graph ---
        self._mem("before backbone")
        feats_list = self.backbone(data_dict['features'], data_dict)
        self._mem("after  backbone")

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # --- 5. Extract Points and Features ---
        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()

        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]

        ref_fps_idx = src_fps_idx = None
        if self.max_superpoints is not None:
            if ref_points_c.shape[0] > self.max_superpoints:
                _, ref_fps_idx = sample_farthest_points(ref_points_c.unsqueeze(0), K=self.max_superpoints)
                ref_fps_idx = ref_fps_idx.squeeze(0)
                ref_points_c = ref_points_c[ref_fps_idx]
            if src_points_c.shape[0] > self.max_superpoints:
                _, src_fps_idx = sample_farthest_points(src_points_c.unsqueeze(0), K=self.max_superpoints)
                src_fps_idx = src_fps_idx.squeeze(0)
                src_points_c = src_points_c[src_fps_idx]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        transform = data_dict['transform'].detach()
    
        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        ref_feats_c = feats_c[:ref_length_c]
        if ref_fps_idx is not None:
            ref_feats_c = ref_feats_c[ref_fps_idx]
        src_feats_c = feats_c[ref_length_c:]
        if src_fps_idx is not None:
            src_feats_c = src_feats_c[src_fps_idx]
        self._mem("before geometric transformer")
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        self._mem("after  geometric transformer")
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices] 
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices] 
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices] 
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices] 
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices] 

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats) 
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)
        self._mem("after  optimal transport")

        output_dict['matching_scores'] = matching_scores

        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        self._mem("end of forward")
        return output_dict


def create_model(config):
    model = GeoTransformer(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
