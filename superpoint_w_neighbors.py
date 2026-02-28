import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

# path to your PLY file
ply_path = "data/UHM_downsampled/train/0.ply"

# indices you provided
target_indices = [75, 411, 2699, 911, 8594, 3380, 6731, 9710, 9633, 119, 
                3441, 6319, 9541, 8732, 6162, 3774, 8296, 3151, 10, 
                7720, 6858, 7409, 7531, 3504, 6937, 4189, 8891, 3721, 
                9241, 2213, 1765, 7547]

# number of nearest neighbors to include for each target (including the target itself)
k_neighbors = 700
# load the PLY using PyVista
mesh = pv.read(ply_path)
# ensure we are working with the point cloud (mesh.points is Nx3)
points = mesh.points.copy()
n_points = points.shape[0]

# build KD-tree and query neighbors
tree = cKDTree(points)
# collect neighbor indices for all targets
all_neighbor_indices = set()
for ti in target_indices:
    # safety: ensure provided index is in range
    if ti < 0 or ti >= n_points:
        raise IndexError(f"target index {ti} out of range (0..{n_points-1})")
    dists, inds = tree.query(points[ti], k=k_neighbors)
    # if k_neighbors == 1, inds is scalar; normalize to iterable
    if np.isscalar(inds):
        inds = [int(inds)]
    all_neighbor_indices.update(map(int, inds))

# prepare RGBA color array (uint8, 0..255) per-point
colors = np.zeros((n_points, 4), dtype=np.uint8)

# translucent gray for background points
gray_rgba = np.array([180, 180, 180, 30], dtype=np.uint8)  # low alpha -> translucent
colors[:] = gray_rgba

# bright red for selected targets + their neighbors
red_rgba = np.array([255, 0, 0, 255], dtype=np.uint8)  # opaque bright red
for idx in all_neighbor_indices:
    colors[idx] = red_rgba

# attach color array to the point cloud
pc = pv.PolyData(points)
pc["RGBA"] = colors   # attaches RGBA as point data

# create plotter and show
p = pv.Plotter()
# render points as spheres so alpha blending and sizes look better
p.add_mesh(pc, scalars="RGBA", rgba=True, point_size=6, render_points_as_spheres=True)
p.add_axes()
p.show_grid()
p.show(title="Highlighted vertices + 25-NN (red) | others translucent gray")
