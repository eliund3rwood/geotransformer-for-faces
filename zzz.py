import torch
import numpy as np
import pyvista as pv
import os

# Import the reader function from your existing script
from neighborhoods_new import read_ply_points

def visualize_pca_reconstruction(pth_path, src_dir, sample_idx=0):
    """
    Reconstructs patches from PCA bases and displays three side-by-side models:
    1. Original point cloud
    2. Reconstructed with ground truth Z
    3. Reconstructed with all-zero Z (the mean patch)
    """
    # 1. Load the saved PCA components and ground truth coefficients
    data = torch.load(pth_path)
    basis = data['basis'].numpy()                 # Shape: [32, 100, 2100]
    mean = data['mean'].numpy()                   # Shape: [32, 2100]
    gt_z = data['gt_z'].numpy()                   # Shape: [32, num_samples, 100]
    patch_indices = data['patch_indices'].numpy() # Shape: [32, 700]
    
    num_superpoints = basis.shape[0]

    # 2. Load the original corresponding point cloud
    filenames = sorted([f for f in os.listdir(src_dir) if f.endswith(".ply")])
    if sample_idx >= len(filenames):
        raise ValueError(f"Sample index {sample_idx} is out of bounds. Only {len(filenames)} files found.")
    
    ply_path = os.path.join(src_dir, filenames[sample_idx])
    original_pts = read_ply_points(ply_path)
    
    # 3. Prepare the reconstructed point clouds
    reconstructed_pts_gt = original_pts.copy()
    reconstructed_pts_zero = original_pts.copy()
    is_patch_point = np.zeros(original_pts.shape[0], dtype=bool)

    # 4. Reconstruct each patch
    for i in range(num_superpoints):
        # Extract data for the specific superpoint
        z_i = gt_z[i, sample_idx]      # [100]
        basis_i = basis[i]             # [100, 2100]
        mean_i = mean[i]               # [2100]
        p_inds = patch_indices[i]
        
        # --- GT Reconstruction (Z * Basis + Mean) ---
        recon_flat_gt = np.dot(z_i, basis_i) + mean_i
        recon_patch_gt = recon_flat_gt.reshape(-1, 3) 
        reconstructed_pts_gt[p_inds] = recon_patch_gt
        
        # --- Zero Reconstruction (0 * Basis + Mean) ---
        # Note: mathematically this is exactly equal to `mean_i`
        z_zero = np.zeros_like(z_i)
        recon_flat_zero = np.dot(z_zero, basis_i) + mean_i
        recon_patch_zero = recon_flat_zero.reshape(-1, 3)
        reconstructed_pts_zero[p_inds] = recon_patch_zero
        
        is_patch_point[p_inds] = True

    # 5. Set up colors for visualization
    # Original (Green Patches)
    colors_orig = np.tile(np.array([200, 200, 200], dtype=np.uint8), (original_pts.shape[0], 1))
    colors_orig[is_patch_point] = [0, 255, 0] 
    
    # GT Reconstruction (Red Patches)
    colors_gt = np.tile(np.array([200, 200, 200], dtype=np.uint8), (reconstructed_pts_gt.shape[0], 1))
    colors_gt[is_patch_point] = [255, 0, 0] 
    
    # Zero Reconstruction (Blue Patches)
    colors_zero = np.tile(np.array([200, 200, 200], dtype=np.uint8), (reconstructed_pts_zero.shape[0], 1))
    colors_zero[is_patch_point] = [0, 0, 255]

    # Create PyVista meshes
    cloud_orig = pv.PolyData(original_pts)
    cloud_orig["rgb"] = colors_orig
    
    cloud_gt = pv.PolyData(reconstructed_pts_gt)
    cloud_gt["rgb"] = colors_gt

    cloud_zero = pv.PolyData(reconstructed_pts_zero)
    cloud_zero["rgb"] = colors_zero

    # 6. Side-by-side PyVista plotter (1 row, 3 columns)
    p = pv.Plotter(shape=(1, 3), window_size=(2000, 700))
    
    # Left Viewport (Original)
    p.subplot(0, 0)
    p.add_text(f"Original Cloud {sample_idx}\n(GT Patches in Green)", font_size=10)
    p.add_points(cloud_orig, scalars="rgb", rgb=True, point_size=3, render_points_as_spheres=True)
    
    # Middle Viewport (GT Reconstruction)
    p.subplot(0, 1)
    p.add_text(f"PCA Reconstruction w/ GT Z\n(Patches in Red)", font_size=10)
    p.add_points(cloud_gt, scalars="rgb", rgb=True, point_size=3, render_points_as_spheres=True)

    # Right Viewport (Zero Reconstruction)
    p.subplot(0, 2)
    p.add_text(f"PCA Reconstruction w/ Zeros (Mean)\n(Patches in Blue)", font_size=10)
    p.add_points(cloud_zero, scalars="rgb", rgb=True, point_size=3, render_points_as_spheres=True)
    
    # Link the cameras so dragging one rotates all of them symmetrically 
    p.link_views()
    p.show()

if __name__ == "__main__":
    visualize_pca_reconstruction(
        pth_path="pca_basis_all.pth",
        src_dir="data/UHM_downsampled/train",
        sample_idx=50 
    )