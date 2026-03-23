import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)

from backbone import KPConvFPN

class CrossAttentionRegressor(nn.Module):
    def __init__(self, feature_dim=256, num_patches=32, num_coeffs=100, nhead=4, num_layers=2):
        super().__init__()
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        
        # 32 learnable tokens
        self.patch_tokens = nn.Parameter(torch.randn(1, num_patches, feature_dim))
        
        # Cross-Attention Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim, 
            nhead=nhead, 
            dim_feedforward=feature_dim * 2,
            batch_first=True,
            norm_first=True 
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final proj from feature_dim -> 100 PCA coeffs
        self.output_proj = nn.Linear(feature_dim, num_coeffs)
        
    def forward(self, src_feats_padded, src_padding_mask):

        # Align tokens to match input batch size
        B = src_feats_padded.shape[0]
        tokens = self.patch_tokens.expand(B, -1, -1)
        
        # Cross Attention
        updated_tokens = self.transformer_decoder(
            tgt=tokens, 
            memory=src_feats_padded,
            memory_key_padding_mask=src_padding_mask
        ) 
        
        # Project to PCA coeffs
        coeffs = self.output_proj(updated_tokens)
        
        return coeffs

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

        self.coeff_regressor = CrossAttentionRegressor(
            feature_dim=1024,
            num_patches=32,
            num_coeffs=100,
            nhead=4,
            num_layers=2
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
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

    def generate_reference_geometry(self, z_delta):
        num_patches = self.patch_indices.shape[0]
        k_neighbors = self.patch_indices.shape[1]

        # Reconstruction: mean + z_delta @ basis
        delta = torch.matmul(z_delta.unsqueeze(1), self.pca_basis).squeeze(1)
        reconstructed_patches_flat = self.pca_mean + delta
        reconstructed_points = reconstructed_patches_flat.view(num_patches, k_neighbors, 3)
        
        # Prepare data for stitching patches together
        flat_indices = self.patch_indices.view(-1).long()
        flat_points = reconstructed_points.contiguous().view(-1, 3)
        num_global_verts = flat_indices.max().item() + 1
        
        # Overlap Logic (Last-Write-Wins)
        patch_scores = torch.arange(num_patches, device=z_delta.device, dtype=torch.long)
        flat_scores = patch_scores.unsqueeze(1).expand(-1, k_neighbors).reshape(-1)
        
        max_scores = torch.full((num_global_verts,), -1, device=z_delta.device, dtype=torch.long)
        max_scores.scatter_reduce_(0, flat_indices, flat_scores, reduce='amax', include_self=False)
        
        is_max_score = flat_scores == max_scores[flat_indices]
        
        valid_flat_positions = torch.arange(flat_indices.size(0), device=z_delta.device)[is_max_score]
        valid_global_indices = flat_indices[is_max_score]
        
        best_idx_per_global = torch.zeros(num_global_verts, dtype=torch.long, device=z_delta.device)
        best_idx_per_global.scatter_(0, valid_global_indices, valid_flat_positions)
        
        ref_points = torch.zeros((num_global_verts, 3), device=z_delta.device, dtype=z_delta.dtype)
        has_points = torch.zeros(num_global_verts, dtype=torch.bool, device=z_delta.device)
        has_points[valid_global_indices] = True
        ref_points[has_points] = flat_points[best_idx_per_global[has_points]]
        
        return ref_points

    def forward(self, data_dict):
        output_dict = {}

        # Isolate src pts
        ref_length_orig = data_dict['lengths'][0][0].item()
        orig_points = data_dict['points'][0] # [ref; src]
        src_points = orig_points[ref_length_orig:]
        src_feats_raw = data_dict['features'][ref_length_orig:]

        # Run backbone on ref + src & extract src coarse feats
        feats_list = self.backbone(data_dict['features'], data_dict)

        coarse_feats_all = feats_list[-1]

        ref_len_c = data_dict['lengths'][-1][0].item()
        src_coarse_feats = coarse_feats_all[ref_len_c:]

        # Predict coeffs via Cross-Attention
        src_feats_batched = src_coarse_feats.unsqueeze(0)
        
        # Padding mask 
        padding_mask = torch.zeros((1, src_coarse_feats.shape[0]), dtype=torch.bool, device=src_coarse_feats.device)
        
        # Pass through regressor
        z_delta_batched = self.coeff_regressor(src_feats_batched, padding_mask)
        
        z_delta = z_delta_batched.squeeze(0)
        output_dict['z_coefficients'] = z_delta

        # Morph ref with pred coeffs 
        new_ref_points = self.generate_reference_geometry(z_delta)
        output_dict['morphed_full'] = new_ref_points 

        # Morph ref with GT coeffs
        gt_z = data_dict['gt_z']
        recon_gt_points = self.generate_reference_geometry(gt_z)
        output_dict['recon_gt_points'] = recon_gt_points

        # Detach for second pass so rigid alignment loss terms (f_loss & c_loss) can't influence pred coeffs
        new_ref_points_detached = new_ref_points.detach()

        # Construct new input for second pass
        new_points = torch.cat([new_ref_points_detached, src_points], dim=0)
        new_lengths = torch.tensor([len(new_ref_points_detached), len(src_points)], dtype=torch.int32)

        data_dict['points'][0] = new_points
        data_dict['lengths'][0] = new_lengths

        orig_feat_dim = data_dict['features'].shape[1]
        new_ref_feats = torch.ones((len(new_ref_points), orig_feat_dim), device=orig_points.device)
        data_dict['features'] = torch.cat([new_ref_feats, src_feats_raw], dim=0)

        feats_list_new = self.backbone(data_dict['features'], data_dict)

        feats_c = feats_list_new[-1]
        feats_f = feats_list_new[0]

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()

        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
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
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
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