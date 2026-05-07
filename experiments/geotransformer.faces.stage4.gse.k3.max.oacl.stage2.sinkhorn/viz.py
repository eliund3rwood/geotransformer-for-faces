"""Visualization utilities for registration results in TensorBoard and ClearML."""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def _to_numpy(x):
    if hasattr(x, 'cpu'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _apply_transform(pts, T):
    """Apply 4x4 rigid transform to Nx3 array."""
    R, t = T[:3, :3], T[:3, 3]
    return pts @ R.T + t


def _subsample(pts, max_pts=3000):
    if pts.shape[0] <= max_pts:
        return pts
    idx = np.random.default_rng(0).choice(pts.shape[0], max_pts, replace=False)
    return pts[idx]


def render_registration_figure(ref_pts, src_pts, estimated_T, title="", max_pts=3000):
    """
    Three-panel figure (XY, XZ, YZ) showing morphed ref overlaid with
    registered src (after applying estimated_T).

    ref_pts, src_pts : Nx3 tensors or arrays
    estimated_T      : 4x4 tensor or array
    Returns          : matplotlib Figure (caller must plt.close it)
    """
    ref_np = _subsample(_to_numpy(ref_pts), max_pts)
    src_np = _subsample(_to_numpy(src_pts), max_pts)
    T_np = _to_numpy(estimated_T)
    src_aligned = _apply_transform(src_np, T_np)

    projs = [('XY', 0, 1), ('XZ', 0, 2), ('YZ', 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (proj, i, j) in zip(axes, projs):
        ax.scatter(ref_np[:, i], ref_np[:, j],
                   c='steelblue', s=0.8, alpha=0.4, label='ref (morphed)', rasterized=True)
        ax.scatter(src_aligned[:, i], src_aligned[:, j],
                   c='tomato', s=0.8, alpha=0.4, label='src (registered)', rasterized=True)
        ax.set_title(proj, fontsize=11)
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(markerscale=6, fontsize=8, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def render_registration_3d(ref_pts, src_pts, estimated_T, title="", max_pts=3000):
    """
    Interactive Plotly 3D scatter of ref and registered src.
    Returns a plotly Figure for logging via ClearML report_plotly.
    """
    ref_np = _subsample(_to_numpy(ref_pts), max_pts)
    src_np = _subsample(_to_numpy(src_pts), max_pts)
    src_aligned = _apply_transform(src_np, _to_numpy(estimated_T))

    fig = go.Figure([
        go.Scatter3d(
            x=ref_np[:, 0], y=ref_np[:, 1], z=ref_np[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='steelblue', opacity=0.5),
            name='ref (morphed)',
        ),
        go.Scatter3d(
            x=src_aligned[:, 0], y=src_aligned[:, 1], z=src_aligned[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='tomato', opacity=0.5),
            name='src (registered)',
        ),
    ])
    fig.update_layout(title=title, scene=dict(aspectmode='data'))
    return fig
