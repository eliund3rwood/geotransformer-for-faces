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

def neighborhoods(src="UHM_downsampled",
                  superpoint_indices=[8748, 2320, 10095, 2103, 8850, 6888, 8985, 8562, 4607, 1860],
                  k_neighbors=700):
    
    results_per_pcd = []
    pcd_paths = []
    col_means_list = []

    for name in os.listdir(src):
        if not name.endswith(".ply"):
            continue
        path = os.path.join(src, name)
        if not os.path.isfile(path):
            continue

        points = read_ply_points(path)
        tree = cKDTree(points)

        rows = []
        for index in superpoint_indices:
            dists, inds = tree.query(points[index], k=k_neighbors)
            neighbor_coords = points[inds]
            rows.append(neighbor_coords.flatten())

        result = np.vstack(rows)        # shape: (num_superpoints, k*3)
        results_per_pcd.append(result)
        pcd_paths.append(path)

    # reorganize --> each 2D array is neighbor coords for each pcd, one array per superpoint
    temp = np.stack(results_per_pcd, axis=0) # (num pcds, num superpoints, k*3)
    arr = [temp[:, r, :] for r in range(temp.shape[1])] # (num superpoints, num pcds, k*3)

    # mean-center columns
    for i in range(len(arr)):
        X = arr[i]
        col_means = X.mean(axis=0)
        col_means_list.append(col_means)
        arr[i] = X-col_means

    return arr, pcd_paths, col_means_list


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
