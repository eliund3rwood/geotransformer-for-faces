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
from backbone import KPConvFPN

# Padding coordinate for src coarse points: far from the (origin-centered) face
# clouds so padded points never enter the geometric-embedding knn of real points.
PAD_COORD = 1.0e4


class CrossAttentionRegressor(nn.Module):
    r"""Batched PCA-coefficient regressor.

    Parameter names/shapes are identical to the single-pair version, so
    checkpoints from the original experiment load directly.
    """

    def __init__(self, feature_dim=256, num_patches=32, num_coeffs=100, nhead=4, num_layers=2,
                 backbone_dim=1024):
        super().__init__()
        self.num_patches = num_patches
        self.feature_dim = feature_dim

        # 32 learnable anatomical patch tokens (queries)
        self.patch_tokens = nn.Parameter(torch.randn(1, num_patches, feature_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 2,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Project pre-transformer backbone src feats (backbone_dim) → feature_dim
        self.feature_proj = nn.Linear(backbone_dim, feature_dim)

        # Project concatenated correspondence pairs (2 * feature_dim) → feature_dim
        self.corr_proj = nn.Linear(feature_dim * 2, feature_dim)

        # Final projection: feature_dim → num_coeffs PCA coefficients per patch
        self.output_proj = nn.Linear(feature_dim, num_coeffs)

    def forward(self, ref_feats_post, src_feats_post, src_feats_backbone,
                ref_corr_feats, src_corr_feats, src_masks=None, corr_masks=None):
        # ref_feats_post:     [B, N_ref_c, feature_dim]   shared-size, no padding
        # src_feats_post:     [B, S_max,   feature_dim]   padded
        # src_feats_backbone: [B, S_max,   backbone_dim]  padded
        # ref/src_corr_feats: [B, Q_max,   feature_dim]   padded
        # src_masks:          [B, S_max] True = valid
        # corr_masks:         [B, Q_max] True = valid
        batch_size, num_ref = ref_feats_post.shape[:2]
        device = ref_feats_post.device

        backbone_proj = self.feature_proj(src_feats_backbone)            # [B, S_max, feature_dim]
        corr_proj = self.corr_proj(
            torch.cat([ref_corr_feats, src_corr_feats], dim=2)           # [B, Q_max, 2*feature_dim]
        )                                                                 # [B, Q_max, feature_dim]

        memory = torch.cat(
            [ref_feats_post, src_feats_post, backbone_proj, corr_proj], dim=1
        )                                                                 # [B, L, feature_dim]

        if src_masks is None:
            src_masks = torch.ones(batch_size, src_feats_post.shape[1], dtype=torch.bool, device=device)
        if corr_masks is None:
            corr_masks = torch.ones(batch_size, ref_corr_feats.shape[1], dtype=torch.bool, device=device)
        ref_pad = torch.zeros(batch_size, num_ref, dtype=torch.bool, device=device)
        # nn.TransformerDecoder convention: True = ignored
        memory_padding_mask = torch.cat([ref_pad, ~src_masks, ~src_masks, ~corr_masks], dim=1)

        tokens = self.patch_tokens.expand(batch_size, -1, -1)            # [B, num_patches, feature_dim]
        updated = self.transformer_decoder(
            tgt=tokens, memory=memory, memory_key_padding_mask=memory_padding_mask
        )                                                                 # [B, num_patches, feature_dim]

        z_delta = self.output_proj(updated)                              # [B, num_patches, num_coeffs]
        return z_delta


class GeoTransformer(nn.Module):
    r"""Shared-reference batched GeoTransformer.

    Differences from the single-pair experiment:
      - data_dict carries a src-only KPConv stack ([src_1..src_B] + lengths) and a
        single canonical `ref_points` tensor; the constant ref graph, partition and
        morph gather indices are built once and cached.
      - backbone runs once on ref and once on the src stack per iteration.
      - the geometric transformer and the coefficient regressor run on padded
        (B, N, C) batches with masks; cheap index logic loops per item.
    """

    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        pca_data = torch.load("pca_basis_all.pth")
        n_comp = cfg.model.num_pca_components

        self.register_buffer("pca_basis", pca_data['basis'][:, :n_comp, :])
        self.register_buffer("pca_mean", pca_data['mean'])
        self.register_buffer("patch_indices", pca_data['patch_indices'])

        self.num_patches = self.patch_indices.shape[0]
        self.num_components = self.pca_basis.shape[1]  # == n_comp

        gt_z_mean = torch.mean(pca_data['gt_z'], dim=1)
        self.register_buffer("z_template", gt_z_mean[:, :n_comp])

        # Morph stitching is z-independent (last-write-wins by patch order over
        # patch_indices), so precompute the gather once: morphing reduces to a
        # batched gather of reconstructed patch points.
        flat_indices = self.patch_indices.view(-1).long()
        num_global_verts = int(flat_indices.max().item()) + 1
        self.num_global_verts = num_global_verts
        k_neighbors = self.patch_indices.shape[1]
        patch_scores = torch.arange(self.num_patches, dtype=torch.long)
        flat_scores = patch_scores.unsqueeze(1).expand(-1, k_neighbors).reshape(-1)
        max_scores = torch.full((num_global_verts,), -1, dtype=torch.long)
        max_scores.scatter_reduce_(0, flat_indices, flat_scores, reduce='amax', include_self=False)
        is_max_score = flat_scores == max_scores[flat_indices]
        valid_flat_positions = torch.arange(flat_indices.size(0))[is_max_score]
        valid_global_indices = flat_indices[is_max_score]
        best_idx_per_global = torch.zeros(num_global_verts, dtype=torch.long)
        best_idx_per_global.scatter_(0, valid_global_indices, valid_flat_positions)
        has_points = torch.zeros(num_global_verts, dtype=torch.bool)
        has_points[valid_global_indices] = True
        self.register_buffer("morph_gather_indices", best_idx_per_global[has_points], persistent=False)
        self.register_buffer("morph_valid_mask", has_points, persistent=False)

        self.coeff_regressor = CrossAttentionRegressor(
            feature_dim=cfg.geotransformer.output_dim,
            num_patches=32,
            num_coeffs=n_comp,
            nhead=4,
            num_layers=2,
            backbone_dim=cfg.geotransformer.input_dim,
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
            cfg.coarse_matching.num_correspondences,
            cfg.coarse_matching.dual_normalization,
            entropy_weighting=cfg.coarse_matching.get('entropy_weighting', False),
            entropy_temperature=cfg.coarse_matching.get('entropy_temperature', 1.0),
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

        # Backbone graph hyperparams — used to build the constant ref graph
        self.num_stages = cfg.backbone.num_stages
        self.init_voxel_size = cfg.backbone.init_voxel_size
        self.init_radius = cfg.backbone.init_radius
        self.neighbor_limits = None  # set by trainer after construction

        # Constant-ref cache, built lazily on the first forward (needs device +
        # neighbor_limits). Plain attributes: kept out of the state dict.
        self.ref_graph = None
        self.ref_feats_input = None
        self.ref_points_raw = None
        self.ref_points_f_base = None
        self.ref_points_c_base = None
        self.ref_node_masks = None
        self.ref_node_knn_indices = None
        self.ref_node_knn_masks = None
        self.fine_ref_idx = None
        self.coarse_ref_idx = None

    @torch.no_grad()
    def _build_ref_cache(self, ref_points):
        r"""Precompute everything that depends only on the constant ref template:
        KPConv graph, point-to-node partition, and morph index lookups."""
        assert self.neighbor_limits is not None, 'Set model.neighbor_limits before the first forward.'
        device = ref_points.device
        lengths = torch.LongTensor([ref_points.shape[0]])
        graph = precompute_data_stack_mode(
            ref_points.cpu(),
            lengths,
            self.num_stages,
            self.init_voxel_size,
            self.init_radius,
            self.neighbor_limits,
        )
        self.ref_graph = {
            key: [t.to(device) for t in graph[key]]
            for key in ['points', 'lengths', 'neighbors', 'subsampling', 'upsampling']
        }
        self.ref_feats_input = torch.ones(ref_points.shape[0], 1, device=device)

        self.ref_points_raw = self.ref_graph['points'][0]
        self.ref_points_f_base = self.ref_graph['points'][1]
        self.ref_points_c_base = self.ref_graph['points'][-1]

        _, self.ref_node_masks, self.ref_node_knn_indices, self.ref_node_knn_masks = point_to_node_partition(
            self.ref_points_f_base, self.ref_points_c_base, self.num_points_in_patch
        )

        # Map fine/coarse ref points to template vertex indices (stable across
        # morphing: morphing displaces vertices but keeps their identity).
        self.fine_ref_idx = torch.cdist(self.ref_points_f_base, self.ref_points_raw).argmin(dim=1)
        self.coarse_ref_idx = torch.cdist(self.ref_points_c_base, self.ref_points_raw).argmin(dim=1)

    def generate_reference_geometry(self, z_delta):
        r"""Reconstruct the morphed template from PCA coefficients.

        Args:
            z_delta: (num_patches, n_comp) or (B, num_patches, n_comp)

        Returns:
            ref_points: (num_global_verts, 3) or (B, num_global_verts, 3)
        """
        squeeze_output = z_delta.dim() == 2
        if squeeze_output:
            z_delta = z_delta.unsqueeze(0)
        batch_size = z_delta.shape[0]

        # (B, P, 1, n) @ (1, P, n, D) -> (B, P, D);  D = k_neighbors * 3
        delta = torch.matmul(z_delta.unsqueeze(2), self.pca_basis.unsqueeze(0)).squeeze(2)
        reconstructed = self.pca_mean.unsqueeze(0) + delta                 # (B, P, D)
        flat_points = reconstructed.view(batch_size, -1, 3)               # (B, P*K, 3)

        ref_points = torch.zeros(
            batch_size, self.num_global_verts, 3, device=z_delta.device, dtype=z_delta.dtype
        )
        ref_points[:, self.morph_valid_mask] = flat_points[:, self.morph_gather_indices]

        if squeeze_output:
            ref_points = ref_points.squeeze(0)
        return ref_points

    def forward(self, data_dict):
        output_dict = {}

        batch_size = int(data_dict['batch_size'])
        device = data_dict['features'].device
        transform = data_dict['transform'].detach()
        if transform.dim() == 2:
            transform = transform.unsqueeze(0)

        if self.ref_graph is None:
            self._build_ref_cache(data_dict['ref_points'])

        # --- 1. Backbone on the shared ref (once per iteration) ---
        ref_feats_list = self.backbone(self.ref_feats_input, self.ref_graph)
        ref_feats_c_pre = ref_feats_list[-1]     # (N_ref_c, backbone_dim)
        ref_feats_f = ref_feats_list[0]          # (N_ref_f, output_dim)

        # --- 2. Backbone on the stacked sources ---
        feats_list = self.backbone(data_dict['features'], data_dict)
        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        lengths_c = data_dict['lengths'][-1].tolist()
        lengths_f = data_dict['lengths'][1].tolist()
        lengths_0 = data_dict['lengths'][0].tolist()

        src_points_c_list = list(torch.split(data_dict['points'][-1].detach(), lengths_c))
        src_points_f_list = list(torch.split(data_dict['points'][1].detach(), lengths_f))
        src_points_list = list(torch.split(data_dict['points'][0].detach(), lengths_0))
        src_feats_c_list = list(torch.split(feats_c, lengths_c))
        src_feats_f_list = list(torch.split(feats_f, lengths_f))

        # --- 3. Train-time per-item src coarse dropout (sparser graphs, like real scans) ---
        if self.training:
            for i in range(batch_size):
                n_src_c = src_points_c_list[i].shape[0]
                keep_frac = torch.empty(1).uniform_(0.4, 1.0).item()
                n_keep = max(int(n_src_c * keep_frac), 24)
                keep_idx = torch.randperm(n_src_c, device=device)[:n_keep].sort().values
                src_points_c_list[i] = src_points_c_list[i][keep_idx]
                src_feats_c_list[i] = src_feats_c_list[i][keep_idx]

        output_dict['src_points_c'] = src_points_c_list
        output_dict['src_points_f'] = src_points_f_list
        output_dict['src_points'] = src_points_list
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f_list

        # --- 4. Per-item src node partition (ref partition is cached) ---
        src_node_masks_list = []
        src_node_knn_indices_list = []
        src_node_knn_masks_list = []
        src_node_knn_points_list = []
        for i in range(batch_size):
            _, node_masks, knn_indices, knn_masks = point_to_node_partition(
                src_points_f_list[i], src_points_c_list[i], self.num_points_in_patch
            )
            padded_points_f = torch.cat(
                [src_points_f_list[i], torch.zeros_like(src_points_f_list[i][:1])], dim=0
            )
            src_node_masks_list.append(node_masks)
            src_node_knn_indices_list.append(knn_indices)
            src_node_knn_masks_list.append(knn_masks)
            src_node_knn_points_list.append(index_select(padded_points_f, knn_indices, dim=0))

        # --- 5. Pad src coarse points/features to the batch max ---
        src_lengths_c = [points.shape[0] for points in src_points_c_list]
        max_src_c = max(src_lengths_c)
        src_points_c_pad = data_dict['points'][-1].new_full((batch_size, max_src_c, 3), PAD_COORD)
        src_feats_c_pad = feats_c.new_zeros(batch_size, max_src_c, feats_c.shape[1])
        src_valid_masks = torch.zeros(batch_size, max_src_c, dtype=torch.bool, device=device)
        for i, n in enumerate(src_lengths_c):
            src_points_c_pad[i, :n] = src_points_c_list[i]
            src_feats_c_pad[i, :n] = src_feats_c_list[i]
            src_valid_masks[i, :n] = True

        # --- 6. Geometric Transformer on the padded batch ---
        # The ref geometric embedding is identical for every batch item, so compute
        # it once at B=1 and expand: the O(N_ref^2 * k * C) embedding intermediates
        # would otherwise be replicated B times (several GiB at this ref size).
        ref_points_c = self.ref_points_c_base
        ref_embeddings = self.transformer.embedding(ref_points_c.unsqueeze(0))     # (1, N, N, C)
        ref_embeddings = ref_embeddings.expand(batch_size, -1, -1, -1)
        src_embeddings = self.transformer.embedding(src_points_c_pad)              # (B, S, S, C)

        ref_feats_c_in = self.transformer.in_proj(ref_feats_c_pre.unsqueeze(0).expand(batch_size, -1, -1))
        src_feats_c_in = self.transformer.in_proj(src_feats_c_pad)
        ref_feats_c_t, src_feats_c_t = self.transformer.transformer(
            ref_feats_c_in,
            src_feats_c_in,
            ref_embeddings,
            src_embeddings,
            masks0=None,
            masks1=~src_valid_masks,   # attention convention: True = ignored
        )
        ref_feats_c_t = self.transformer.out_proj(ref_feats_c_t)
        src_feats_c_t = self.transformer.out_proj(src_feats_c_t)
        ref_feats_c_norm = F.normalize(ref_feats_c_t, p=2, dim=2)         # (B, N_ref_c, C)
        src_feats_c_norm = F.normalize(src_feats_c_t, p=2, dim=2)         # (B, S_max, C)
        src_feats_c_norm_list = [src_feats_c_norm[i, :n] for i, n in enumerate(src_lengths_c)]

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm_list

        # --- 7. Coarse matching per item (predicted correspondences) ---
        ref_node_corr_indices_list = []
        src_node_corr_indices_list = []
        node_corr_scores_list = []
        with torch.no_grad():
            for i in range(batch_size):
                ref_idx, src_idx, scores = self.coarse_matching(
                    ref_feats_c_norm[i], src_feats_c_norm_list[i],
                    self.ref_node_masks, src_node_masks_list[i],
                )
                ref_node_corr_indices_list.append(ref_idx)
                src_node_corr_indices_list.append(src_idx)
                node_corr_scores_list.append(scores)
        output_dict['ref_node_corr_indices'] = list(ref_node_corr_indices_list)
        output_dict['src_node_corr_indices'] = list(src_node_corr_indices_list)

        # --- 8. Coefficient regression (batched, padded correspondence features) ---
        corr_lengths = [idx.shape[0] for idx in ref_node_corr_indices_list]
        max_corr = max(corr_lengths)
        feat_dim = ref_feats_c_norm.shape[2]
        ref_corr_feats_pad = ref_feats_c_norm.new_zeros(batch_size, max_corr, feat_dim)
        src_corr_feats_pad = ref_feats_c_norm.new_zeros(batch_size, max_corr, feat_dim)
        corr_valid_masks = torch.zeros(batch_size, max_corr, dtype=torch.bool, device=device)
        for i in range(batch_size):
            n = corr_lengths[i]
            ref_corr_feats_pad[i, :n] = ref_feats_c_norm[i, ref_node_corr_indices_list[i]]
            src_corr_feats_pad[i, :n] = src_feats_c_norm_list[i][src_node_corr_indices_list[i]]
            corr_valid_masks[i, :n] = True

        # z_delta carries gradients; morph_loss backprops through here into transformer+backbone
        z_delta = self.coeff_regressor(
            ref_feats_c_norm,
            src_feats_c_norm,
            src_feats_c_pad,
            ref_corr_feats_pad,
            src_corr_feats_pad,
            src_masks=src_valid_masks,
            corr_masks=corr_valid_masks,
        )   # (B, num_patches, n_comp)
        output_dict['z_coefficients'] = z_delta

        # Gradient-enabled morphed geometry for the dense correspondence loss.
        output_dict['morphed_full_grad'] = self.generate_reference_geometry(z_delta)   # (B, V, 3)

        # --- 9. Morph ref per item, then GT correspondences ---
        gt_z = data_dict['gt_z']
        if gt_z.dim() == 2:
            gt_z = gt_z.unsqueeze(0)
        gt_z = gt_z[:, :, :self.num_components]

        with torch.no_grad():
            output_dict['recon_gt_points'] = self.generate_reference_geometry(gt_z)

            morphed_ref_full = self.generate_reference_geometry(z_delta.detach())      # (B, V, 3)
            output_dict['morphed_full'] = morphed_ref_full

            ref_points_f_b = morphed_ref_full[:, self.fine_ref_idx]                    # (B, N_ref_f, 3)
            ref_points_c_b = morphed_ref_full[:, self.coarse_ref_idx]                  # (B, N_ref_c, 3)
            output_dict['ref_points_c'] = ref_points_c_b
            output_dict['ref_points_f'] = ref_points_f_b
            output_dict['ref_points'] = morphed_ref_full

            ref_padded_points_f = torch.cat(
                [ref_points_f_b, ref_points_f_b.new_zeros(batch_size, 1, 3)], dim=1
            )
            ref_node_knn_points_b = ref_padded_points_f[:, self.ref_node_knn_indices]  # (B, N_ref_c, K, 3)

            gt_node_corr_indices_list = []
            gt_node_corr_overlaps_list = []
            for i in range(batch_size):
                gt_indices, gt_overlaps = get_node_correspondences(
                    ref_points_c_b[i],
                    src_points_c_list[i],
                    ref_node_knn_points_b[i],
                    src_node_knn_points_list[i],
                    transform[i],
                    self.matching_radius,
                    ref_masks=self.ref_node_masks,
                    src_masks=src_node_masks_list[i],
                    ref_knn_masks=self.ref_node_knn_masks,
                    src_knn_masks=src_node_knn_masks_list[i],
                )
                gt_node_corr_indices_list.append(gt_indices)
                gt_node_corr_overlaps_list.append(gt_overlaps)
            output_dict['gt_node_corr_indices'] = gt_node_corr_indices_list
            output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps_list

            if self.training:
                for i in range(batch_size):
                    if gt_node_corr_indices_list[i].shape[0] == 0:
                        continue  # no GT overlap: fall back to predicted correspondences
                    ref_idx, src_idx, scores = self.coarse_target(
                        gt_node_corr_indices_list[i], gt_node_corr_overlaps_list[i]
                    )
                    if ref_idx.shape[0] == 0:
                        continue
                    ref_node_corr_indices_list[i] = ref_idx
                    src_node_corr_indices_list[i] = src_idx
                    node_corr_scores_list[i] = scores

        # --- 10. Fine matching on the morphed ref (items concatenated along dim 0) ---
        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)

        ref_knn_points_chunks = []
        src_knn_points_chunks = []
        ref_knn_masks_chunks = []
        src_knn_masks_chunks = []
        ref_knn_feats_chunks = []
        src_knn_feats_chunks = []
        fine_chunks = []
        start = 0
        for i in range(batch_size):
            ref_idx = ref_node_corr_indices_list[i]
            src_idx = src_node_corr_indices_list[i]

            ref_knn_indices_i = self.ref_node_knn_indices[ref_idx]                     # (P, K)
            src_knn_indices_i = src_node_knn_indices_list[i][src_idx]                  # (P, K)

            ref_knn_points_chunks.append(ref_node_knn_points_b[i][ref_idx])
            src_knn_points_chunks.append(src_node_knn_points_list[i][src_idx])
            ref_knn_masks_chunks.append(self.ref_node_knn_masks[ref_idx])
            src_knn_masks_chunks.append(src_node_knn_masks_list[i][src_idx])

            ref_knn_feats_chunks.append(index_select(ref_padded_feats_f, ref_knn_indices_i, dim=0))
            src_padded_feats_f = torch.cat(
                [src_feats_f_list[i], torch.zeros_like(src_feats_f_list[i][:1])], dim=0
            )
            src_knn_feats_chunks.append(index_select(src_padded_feats_f, src_knn_indices_i, dim=0))

            end = start + ref_idx.shape[0]
            fine_chunks.append((start, end))
            start = end

        ref_node_corr_knn_points = torch.cat(ref_knn_points_chunks, dim=0)
        src_node_corr_knn_points = torch.cat(src_knn_points_chunks, dim=0)
        ref_node_corr_knn_masks = torch.cat(ref_knn_masks_chunks, dim=0)
        src_node_corr_knn_masks = torch.cat(src_knn_masks_chunks, dim=0)
        ref_node_corr_knn_feats = torch.cat(ref_knn_feats_chunks, dim=0)
        src_node_corr_knn_feats = torch.cat(src_knn_feats_chunks, dim=0)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks
        output_dict['fine_chunks'] = fine_chunks
        output_dict['node_corr_counts'] = torch.LongTensor([end - start for start, end in fine_chunks])

        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(
            matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks
        )

        output_dict['matching_scores'] = matching_scores

        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points_list = []
            src_corr_points_list = []
            corr_scores_list = []
            estimated_transform_list = []
            for i, (chunk_start, chunk_end) in enumerate(fine_chunks):
                ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                    ref_node_corr_knn_points[chunk_start:chunk_end],
                    src_node_corr_knn_points[chunk_start:chunk_end],
                    ref_node_corr_knn_masks[chunk_start:chunk_end],
                    src_node_corr_knn_masks[chunk_start:chunk_end],
                    matching_scores[chunk_start:chunk_end],
                    node_corr_scores_list[i],
                )
                ref_corr_points_list.append(ref_corr_points)
                src_corr_points_list.append(src_corr_points)
                corr_scores_list.append(corr_scores)
                estimated_transform_list.append(estimated_transform)

            output_dict['ref_corr_points'] = ref_corr_points_list
            output_dict['src_corr_points'] = src_corr_points_list
            output_dict['corr_scores'] = corr_scores_list
            output_dict['estimated_transform'] = torch.stack(estimated_transform_list, dim=0)  # (B, 4, 4)

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
