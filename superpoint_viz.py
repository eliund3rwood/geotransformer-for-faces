import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

# path to your PLY file
ply_path = "data/UHM_downsampled/train/0.ply"

# The point you want to see the 700 neighbors for
focal_index = 6731

# Your 32 superpoint indices
target_indices = [75, 411, 2699, 911, 8594, 3380, 6731, 9710, 9633, 119, 
                3441, 6319, 9541, 8732, 6162, 3774, 8296, 3151, 10, 
                7720, 6858, 7409, 7531, 3504, 6937, 4189, 8891, 3721, 
                9241, 2213, 1765, 7547]

# Load mesh
mesh = pv.read(ply_path)
points = mesh.points

# 1. Find the 700 nearest neighbors for the focal index
tree = cKDTree(points)
_, neighbor_indices = tree.query(points[focal_index], k=700)

# 2. Extract specific point sets for plotting
neighbor_patch = points[neighbor_indices]
superpoints = points[target_indices]
labels = [str(idx) for idx in target_indices]

# Setup Plotter
p = pv.Plotter()

# 3. Background: The whole mesh (Very faint)
p.add_mesh(mesh, color="gray", opacity=0.25, render_points_as_spheres=True, point_size=2)

# 4. Neighbors: The 700-point patch (Yellow)
p.add_mesh(neighbor_patch, color="yellow", point_size=4, 
           render_points_as_spheres=True, label=f"Neighbors of {focal_index}")

# 5. Superpoints: The 32 key indices (Red)
# We plot these on top so they aren't hidden by the yellow patch
p.add_mesh(superpoints, color="red", point_size=12, 
           render_points_as_spheres=True, name="superpoints")

# 6. Labels: For all 32 superpoints
p.add_point_labels(superpoints, labels, font_size=14, 
                   point_color="red", text_color="white", 
                   shadow=True, always_visible=True)

p.add_axes()
p.add_legend()
p.show(title=f"Inspecting Superpoint {focal_index} and its 700 Neighbors")