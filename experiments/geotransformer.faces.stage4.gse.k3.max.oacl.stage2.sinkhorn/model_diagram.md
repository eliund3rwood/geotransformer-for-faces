# Model Architecture Diagram

```
data_dict['points'][0], data_dict['lengths'][0]   (flat, pre-merge ref+src)
        │
        │  split at ref_length_stage0
        ▼
  src_points_raw  [N_src, 3]
        │
        │  FPS → 1024 pts, center by centroid
        ▼
  src_coords_centered  [1, 1024, 3]
        │
        ▼
 CrossAttentionRegressor
  ├─ PointNetPPEncoder
  │    FPS (num_sampled_points centroids)
  │    → kNN per centroid → relative coords
  │    → shared MLP → max-pool over k neighbors
  │    → [1, num_sampled_points, 128]
  │
  ├─ patch_tokens  [1, 32, 128]  (learnable)
  │    cross-attend to encoded point feats
  │    (TransformerDecoder, 2 layers)
  │    → updated_tokens  [1, 32, 128]
  │
  └─ output_proj  Linear(128, 101)
       → [1, 32, 101]
        │
        ├─ col 0 → scale_logits, mean over 32 patches
        │          pred_scale = 0.4 + 1.2·sigmoid(·)  [scalar]
        │
        └─ cols 1:101 → z_delta  [32, 100]

        │                              │
        ▼                              ▼
  src_points_raw                generate_reference_geometry(z_delta.detach())
  / pred_scale.detach()           reconstructed = pca_mean + z_delta @ pca_basis
  → src_points_scaled             patch stitching (last-write-wins per vertex)
                                  → new_ref_points  [V, 3]
        │                              │
        └──────────────┬───────────────┘
                       │  cat([new_ref_points, src_points_scaled])
                       ▼
          precompute_data_stack_mode()     ← runs on CPU, result moved to GPU
          (voxel downsample + KNN graph, 4 stages)
                       │
                       ▼
              KPConvFPN backbone
          feats_list[0]  → fine feats_f    (stage 1)
          feats_list[-1] → coarse feats_c  (stage 4)
                       │
          ┌────────────┴─────────────┐
          ▼                          ▼
   feats_f, points_f           feats_c, points_c
   [ref | src split]           [ref | src split]
                                     │
                          optional FPS to max_superpoints
                          (ref_points_c, src_points_c)
                                     │
                          point_to_node_partition
                          (assigns fine pts to superpoints,
                           builds knn_indices/masks per node)
                                     │
                          get_node_correspondences
                          → gt_node_corr_indices/overlaps
                                     │
          ┌──────────────────────────┤
          │                          ▼
          │              GeometricTransformer
          │              (self-attn with RPE geometry
          │               + cross-attn ref ↔ src)
          │              → ref_feats_c_norm  [N_ref_c, D]
          │              → src_feats_c_norm  [N_src_c, D]
          │                          │
          │              SuperPointMatching
          │              (dual-norm top-k coarse correspondences)
          │                          │
          │         [training: SuperPointTargetGenerator
          │          replaces predicted with GT correspondences]
          │                          │
          └──────────────┬───────────┘
                         │  index fine feats & points
                         │  at matched superpoint patches
                         ▼
             ref_node_corr_knn_feats  [C, K, D]
             src_node_corr_knn_feats  [C, K, D]
                         │
                         │  dot-product scores / sqrt(D)
                         ▼
          LearnableLogOptimalTransport
          (Sinkhorn on fine patch feature scores)
          → matching_scores  [C, K+1, K+1]
                         │
                         ▼
          LocalGlobalRegistration
          (weighted SVD → estimated rigid transform)
                         │
                         ▼
                   output_dict
        (estimated_transform, ref/src_corr_points,
         matching_scores, gt_node_corr_indices,
         z_coefficients, pred_scale,
         morphed_full, recon_gt_points, ...)
```
