"""
Plot registration results for standalone .npy point clouds vs. the morphed template.

Usage (inside Docker):
    python plot_registration.py --snapshot /path/to/snapshot.pth.tar \
        --srcs 2189.npy plank.npy plank_scaled.npy \
        --out_dir ./registration_plots

The script:
  1. Loads the model from the given snapshot.
  2. For each src .npy (Nx3 float array), runs inference using the
     PCA-mean face as the initial ref placeholder.
  3. Plots morphed-ref vs. registered-src in three projections.
  4. Saves one PNG per input file.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from config import make_cfg
from model import create_model
from viz import render_registration_figure

EXP_DIR = os.path.dirname(os.path.realpath(__file__))


def _load_snapshot(model, snapshot_path):
    state = torch.load(snapshot_path, map_location='cpu')
    # Trainer saves under 'model' key; handle both formats
    sd = state.get('model', state)
    model.load_state_dict(sd, strict=True)
    print(f"Loaded snapshot: {snapshot_path}")


def _build_data_dict(ref_pts_np, src_pts_np, gt_z_np, device):
    """Build the flat data_dict expected by model.forward() (batch_size=1,
    precompute_data=False format)."""
    ref_t = torch.from_numpy(ref_pts_np.astype(np.float32)).to(device)
    src_t = torch.from_numpy(src_pts_np.astype(np.float32)).to(device)

    points = torch.cat([ref_t, src_t], dim=0)
    lengths = torch.tensor([ref_t.shape[0], src_t.shape[0]], dtype=torch.long)

    n_total = points.shape[0]
    features = torch.ones((n_total, 1), dtype=torch.float32, device=device)

    transform = torch.eye(4, dtype=torch.float32, device=device)

    gt_z = torch.from_numpy(gt_z_np.astype(np.float32)).to(device)

    return {
        'points': points,
        'lengths': lengths.to(device),
        'features': features,
        'transform': transform,
        'gt_z': gt_z,
        'batch_size': 1,
    }


def run_inference(model, ref_pts_np, src_pts_np, gt_z_np, device):
    data_dict = _build_data_dict(ref_pts_np, src_pts_np, gt_z_np, device)
    with torch.no_grad():
        output_dict = model(data_dict)
    return output_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True,
                        help='Path to model snapshot (.pth.tar)')
    parser.add_argument('--srcs', nargs='+',
                        default=['2189.npy', 'plank.npy', 'plank_scaled.npy'],
                        help='Source point cloud .npy files (Nx3)')
    parser.add_argument('--out_dir', default='./registration_plots',
                        help='Directory to write output PNGs')
    parser.add_argument('--neighbor_limits', nargs='+', type=int, default=None,
                        help='Neighbor limits for graph building (optional, calibrated from train if omitted)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.chdir(EXP_DIR)

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"Device: {device}")

    cfg = make_cfg()
    model = create_model(cfg).to(device).eval()
    _load_snapshot(model, args.snapshot)

    if args.neighbor_limits is not None:
        model.neighbor_limits = args.neighbor_limits
    else:
        # Calibrate from a small subset of training data
        print("Calibrating neighbor limits from training data ...")
        from dataset import train_valid_data_loader
        from geotransformer.utils.data import (
            calibrate_neighbors_stack_mode,
            registration_collate_fn_stack_mode,
        )
        import torch.utils.data
        from geotransformer.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset

        train_dataset = ThreeDMatchPairDataset(
            cfg.data.dataset_root,
            'train',
            point_limit=cfg.train.point_limit,
            use_augmentation=False,
        )
        neighbor_limits = calibrate_neighbors_stack_mode(
            train_dataset,
            registration_collate_fn_stack_mode,
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
        )
        model.neighbor_limits = neighbor_limits
        print(f"  neighbor_limits = {neighbor_limits}")

    # Use the PCA template mean as the initial ref placeholder.
    # model.pca_mean is the flattened per-patch mean; reshape to get 3D points.
    pca_data = torch.load(os.path.join(EXP_DIR, 'pca_basis_all.pth'), map_location='cpu')
    patch_indices = pca_data['patch_indices']            # [P, K]
    pca_mean_flat = pca_data['mean']                     # [P*K*3] or [P, K*3]

    # Reconstruct a dense template by averaging over patches
    P, K = patch_indices.shape
    mean_patches = pca_mean_flat.view(P, K, 3)           # [P, K, 3]
    flat_idx = patch_indices.view(-1).long()             # [P*K]
    flat_pts = mean_patches.contiguous().view(-1, 3)     # [P*K, 3]
    num_verts = flat_idx.max().item() + 1
    template_np = np.zeros((num_verts, 3), dtype=np.float32)
    count = np.zeros(num_verts, dtype=np.float32)
    for flat_i in range(flat_idx.shape[0]):
        v = flat_idx[flat_i].item()
        template_np[v] += flat_pts[flat_i].numpy()
        count[v] += 1
    mask = count > 0
    template_np[mask] /= count[mask, None]

    # gt_z: use all-zero coefficients (mean face in PCA space)
    gt_z_np = np.zeros((P, 100), dtype=np.float32)

    for src_path in args.srcs:
        if not os.path.isabs(src_path):
            src_path_full = os.path.join(EXP_DIR, src_path)
        else:
            src_path_full = src_path

        if not os.path.exists(src_path_full):
            print(f"[skip] not found: {src_path_full}")
            continue

        src_np = np.load(src_path_full).astype(np.float32)
        if src_np.ndim != 2 or src_np.shape[1] != 3:
            print(f"[skip] unexpected shape {src_np.shape} in {src_path}")
            continue

        name = os.path.splitext(os.path.basename(src_path))[0]
        print(f"Processing {name}  ({src_np.shape[0]} pts) ...")

        try:
            output_dict = run_inference(model, template_np, src_np, gt_z_np, device)
        except Exception as e:
            print(f"  [error] {e}")
            import traceback; traceback.print_exc()
            continue

        ref_pts = output_dict['ref_points'].detach().cpu()
        src_pts = output_dict['src_points'].detach().cpu()
        T_est  = output_dict['estimated_transform'].detach().cpu()
        pred_scale = output_dict['pred_scale'].detach().cpu().item()

        fig = render_registration_figure(
            ref_pts, src_pts, T_est,
            title=f'{name}  (pred_scale={pred_scale:.3f})',
        )

        out_path = os.path.join(args.out_dir, f'{name}_registration.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print("Done.")


if __name__ == '__main__':
    main()
