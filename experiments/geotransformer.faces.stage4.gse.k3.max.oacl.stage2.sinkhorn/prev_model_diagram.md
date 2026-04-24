# Model Architecture Diagram

```
Raw Source Points (flat tensor)
        │
        ▼
  FPS subsample to 1024 pts
  center by centroid
        │
        ▼
 CrossAttentionRegressor
  (MLP encode → cross-attn
   with 32 learnable tokens)
        │
        ├──────────────────────┐
        ▼                      ▼
  pred_scale [scalar]    z_delta [32, 100]
        │                      │
        │               generate_reference_geometry()
        │               (PCA decode → morph ref mesh)
        │                      │
        ▼                      ▼
  src /= pred_scale      morphed_ref_points
        │                      │
        └──────────┬───────────┘
                   ▼
        precompute_data_stack_mode()
        (build KNN graph, 4 stages)
                   │
                   ▼
            KPConvFPN backbone
        (4-stage encoder, produces
         fine feats [stage 1]
         coarse feats [stage 3])
                   │
          ┌────────┴────────┐
          ▼                 ▼
   feats_f [fine]     feats_c [coarse]
   points_f           points_c  ← superpoints
          │                 │
          │     GeometricTransformer
          │     (self-attn with RPE geometry
          │      + cross-attn ref↔src)
          │                 │
          │          ref_feats_c_norm
          │          src_feats_c_norm
          │                 │
          │       SuperPointMatching
          │    (top-k coarse correspondences)
          │                 │
          │    [training: SuperPointTargetGenerator
          │     replaces with GT correspondences]
          │                 │
          └────────┬────────┘
                   ▼
        index fine feats & points
        at matched superpoint patches
                   │
                   ▼
       LearnableLogOptimalTransport
         (Sinkhorn on fine patch
          feature dot-products)
                   │
                   ▼
        LocalGlobalRegistration
        (weighted SVD → estimated
         rigid transform)
                   │
                   ▼
            output_dict
     (transform, correspondences,
      z_coefficients, pred_scale, ...)
```
