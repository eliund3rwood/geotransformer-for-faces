import numpy as np
import os
from plyfile import PlyData
from scipy.spatial import cKDTree
import pyvista as pv

def read_ply_points(path):
    ply = PlyData.read(path)
    vertex = ply['vertex']
    points = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T.astype(np.float64)
    return points

def neighborhoods(src="data/UHM_downsampled/train",
                  superpoint_indices=[75, 411, 2699, 911, 8594, 3380, 6731, 9710, 9633, 119, 
                                      3441, 6319, 9541, 8732, 6162, 3774, 8296, 3151, 10, 
                                      7720, 6858, 7409, 7531, 3504, 6937, 4189, 8891, 3721, 
                                      9241, 2213, 1765, 7547],
                  k_neighbors=700):
    
    results_per_pcd = []
    pcd_paths = []
    col_means_list = []
    
    filenames = sorted([f for f in os.listdir(src) if f.endswith(".ply")])
    
    # Calculate neighbor indices once (using Face 1)
    template_path = os.path.join(src, filenames[0])
    template_points = read_ply_points(template_path)
    template_tree = cKDTree(template_points)

    fixed_indices_list = []
    for index in superpoint_indices:
        _, inds = template_tree.query(template_points[index], k=k_neighbors)
        fixed_indices_list.append(inds)
    fixed_indices_array = np.stack(fixed_indices_list) # Shape: (32, 700)

    # Apply to all faces
    for name in filenames:
        path = os.path.join(src, name)
        if not os.path.isfile(path):
            continue

        points = read_ply_points(path)

        rows = []
        for i in range(len(superpoint_indices)):
            inds = fixed_indices_array[i]
            neighbor_coords = points[inds]
            rows.append(neighbor_coords.flatten())

        result = np.vstack(rows)
        results_per_pcd.append(result)
        pcd_paths.append(path)

    # Reorganize to (num superpoints, num pcds, k*3)
    temp = np.stack(results_per_pcd, axis=0)
    arr = [temp[:, r, :] for r in range(temp.shape[1])]

    # Mean-center
    for i in range(len(arr)):
        X = arr[i]
        col_means = X.mean(axis=0)
        col_means_list.append(col_means)
        arr[i] = X - col_means
    
    return arr, pcd_paths, col_means_list, fixed_indices_array

  
def visualize_perturbed_pcd(src, superpoint, perturbed_neighbors, point_size=3):
    pts = read_ply_points(src)
    N = pts.shape[0]
    k = perturbed_neighbors.shape[0]
    tree = cKDTree(pts)
    dists, inds = tree.query(pts[superpoint], k=k)

    base_color = np.tile(np.array([200,200,200], dtype=np.uint8), (N,1))
    colors_base = base_color.copy()
    colors_base[inds] = np.array([255,0,0], dtype=np.uint8)  # original neighbors red

    cloud = pv.PolyData(pts)
    cloud["rgb"] = colors_base

    p = pv.Plotter(window_size=(1100,800))
    p.add_points(cloud, scalars="rgb", rgb=True, point_size=point_size, render_points_as_spheres=True)

    neigh_cloud = pv.PolyData(perturbed_neighbors)
    neigh_cloud["rgb"] = np.tile(np.array([0,0,255], dtype=np.uint8), (k,1))  # perturbed neighbors blue
    p.add_points(neigh_cloud, scalars="rgb", rgb=True, point_size=point_size, render_points_as_spheres=True)

    p.add_axes()
    p.show()
