# Shared-ref batched GeoTransformer (faces)

Batched (B > 1) variant of `geotransformer.faces.stage4.gse.k3.max.oacl.stage2.sinkhorn`,
exploiting the fact that the reference is always the same average-face template.

## Design

**Data layout** (`dataset.py`): the collate stacks only the sources
(`[src_1..src_B]` + `lengths`) and precomputes the KPConv graph in the dataloader
workers (the original experiment rebuilt the joint graph on CPU inside
`model.forward()` every iteration). The canonical ref is passed once per batch as
`ref_points`. Fixed-shape per-sample tensors (`transform`, `gt_z`, `morphed_full`,
`gt_scale`) are stacked along a batch dim.

**Constant-ref cache** (`model.py::_build_ref_cache`): the ref KPConv graph,
point-to-node partition, fine/coarse vertex index lookups are computed once on the
first forward (needs `model.neighbor_limits`, set by the trainer). The morph
"last-write-wins" stitching indices are z-independent and precomputed in
`__init__`, so morphing is a batched gather.

**Per-iteration flow**:
1. backbone on ref (once per iteration, not once per sample; gradients flow),
2. backbone on the src stack,
3. per-item coarse dropout (train) and node partition (cheap loops, `no_grad`),
4. geometric transformer + coefficient regressor on padded `(B, N, C)` batches
   with masks (src padded at coordinate `1e4` so padding never enters the
   geometric-embedding knn of real points),
5. coarse matching / GT correspondences / LGR loop per item (`no_grad` index logic),
6. fine matching + Sinkhorn run on all items' node correspondences concatenated
   along dim 0 (one OT call per batch).

**Losses** (`loss.py`): coarse circle loss averaged per item; fine loss operates on
the concatenated stack with per-item transforms via `repeat_interleave`; morph/dense
losses are natively batched. Evaluator averages metrics across batch items.

**Ref geometric embedding** is computed once at B=1 and expanded across the batch
(the `(B, N_ref, N_ref, k, C)` embedding intermediate is ~5 GiB per extra batch
item at this ref size — at B=4 it OOMed a 24 GB card before this change).

## Differences vs. the single-pair experiment

- **Per-cloud GroupNorm** (`backbone.py`): the library GroupNorm pools statistics
  over the whole stacked point set, so the original model couples ref and src
  features through the norm stats. With batching that would make every sample
  depend on its batchmates, so the backbone here normalizes each cloud separately
  (parameter names unchanged — old checkpoints load; `compare_with_original.py`
  measured the resulting drift on a trained checkpoint: z within 5e-3, estimated
  transforms within 1e-3).
- Rotation augmentation is applied to **src only** (the original randomly rotated
  ref or src). The ref stays canonical so it can be shared across the batch and
  stays consistent with the PCA morph space. Ref noise augmentation is dropped for
  the same reason.
- `model.neighbor_limits` must be set before the first forward (trainval does it).
- `output_dict` per-item entries (`src_points_c`, `ref/src_node_corr_indices`,
  `gt_node_corr_*`, `ref/src_corr_points`, `corr_scores`, `src_feats_c`) are lists
  of length B; `ref_points_c/f`, `z_coefficients`, `morphed_full*`,
  `estimated_transform` are batched tensors. Fine-stage stacks carry
  `fine_chunks` / `node_corr_counts` for per-item slicing.
- Parameter names are unchanged → checkpoints from the original experiment load
  directly (`load_state_dict`).

## Validation & performance (RTX 4090, val samples, trained checkpoint)

- `sanity_check.py`: B=2 vs B=1 per-item outputs agree to float noise
  (max |dz| ≤ 2.6e-3, transforms ≤ 5e-4, losses to ~1e-4); train-mode backward
  produces finite grads for all 300 parameters.
- `benchmark.py` (train mode, fwd+bwd): batched B=4 **239 ms/iter (60 ms/sample,
  peak 14.8 GiB)** vs original 4×B=1 **861 ms (215 ms/sample, peak 4.8 GiB)**
  → **3.6× faster per sample**.

## Usage

All scripts assume the geotransformer docker image (see repo makefile), run from
this directory:

```bash
python trainval.py                 # train with config.py (mirrors original config.py)
python trainval_downsampled.py     # train with config_dowsampled.py (mirrors the
                                   #   .vl.newdata runs: voxel 0.025, no rotation aug,
                                   #   morph/dense loss x100, 180 epochs), batch_size 4
python sanity_check.py             # B=2 vs B=1 equivalence + backward smoke test
python compare_with_original.py    # drift vs original implementation (informational)
python benchmark.py [--batch-size N] [--skip-original]   # throughput / memory probing
```

Gotcha inherited from upstream: `random_sample_rotation(factor)` divides by the
factor, so `augmentation_rotation = 0.0` (as in the downsampled config) produces
NaN rotations unless guarded — the dataset guards it, like the original did.
Edit `_C.exp_name` in `config_dowsampled.py` per run, as in the original.
