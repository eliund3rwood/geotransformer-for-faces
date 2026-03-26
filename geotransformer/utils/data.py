from functools import partial

import numpy as np
import torch
from sklearn.decomposition import PCA
from geotransformer.modules.ops import grid_subsample, radius_search
from geotransformer.utils.torch import build_dataloader

TEMPLATE_CACHE = {
    'r_ref': None,
    'v_anchor': None
}

# BUFFER-X helpers

def bufferx_radius(pts, target_pct=5.0):
    """Calculates density-aware radius using torch"""
    # pts: [N, 3] torch.Tensor
    N = pts.shape[0]
    num_kpts = min(N, 1000)
    
    # Randomly sample query points
    indices = torch.randperm(N, device=pts.device)[:num_kpts]
    kpts = pts[indices]
    
    # Compute squared distances [num_kpts, N]
    dist_sqr = torch.cdist(kpts, pts).pow(2)
    
    low, high = 0.0, 500.0 
    for _ in range(40):
        r = (low + high) / 2.0
        # Average percentage of points found within radius r
        avg_pct = (dist_sqr < r**2).float().mean() * 100
        
        if avg_pct < target_pct:
            low = r
        else:
            high = r
    return float(r)

def bufferx_voxel(pts):
    """Determines adaptive voxel size using torch PCA """
    # pts: [N, 3] torch.Tensor
    N = pts.shape[0]
    
    # Center the points
    pts_mean = pts.mean(dim=0)
    centered_pts = pts - pts_mean
    
    # Sample subset for Covariance calculation (efficient PCA) 
    num_samples = min(N, 2000)
    indices = torch.randperm(N, device=pts.device)[:num_samples]
    sampled = centered_pts[indices]
    
    # Compute Covariance Matrix
    cov = torch.mm(sampled.t(), sampled) / (num_samples - 1)
    
    # Eigen Decomposition (ascending order: evals[0] is lambda_3) 
    evals, evecs = torch.linalg.eigh(cov)
    
    # Sphericity (lambda_3 / lambda_1)
    sphericity = evals[0] / evals[2]
    
    # Spread along the smallest eigenvector v_3 
    v3 = evecs[:, 0]
    proj = torch.mv(pts, v3)
    z_range = proj.max() - proj.min()
    
    # Adaptive coefficient
    alpha = 1.0 if sphericity < 0.05 else 1.5
    v = (torch.sqrt(z_range) / 100.0) * alpha
    
    return float(torch.clamp(v, min=0.001))

# Stack mode utilities


def precompute_data_stack_mode(points, lengths, num_stages, voxel_size, radius, neighbor_limits):

    points = points.cpu()
    lengths = lengths.cpu()

    fixed_indices = torch.tensor([75, 411, 2699, 911, 8594, 3380, 6731, 9710, 9633, 119, 
                                  3441, 6319, 9541, 8732, 6162, 3774, 8296, 3151, 10, 
                                  7720, 6858, 7409, 7531, 3504, 6937, 4189, 8891, 3721, 
                                  9241, 2213, 1765, 7547], dtype=torch.long, device=points.device)

    assert num_stages == len(neighbor_limits)

    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= 2

    original_coarse_ref_points_length = lengths_list[-1][0]
    original_coarse_src_points_length = lengths_list[-1][1]

    new_coarse_points_length = fixed_indices.numel() + original_coarse_src_points_length
    new_coarse_points = torch.cat(
        [points_list[0][fixed_indices], points_list[-1][original_coarse_ref_points_length:]], dim=0
    )
    points_list[-1] = new_coarse_points
    lengths_list[-1][0] = torch.tensor([fixed_indices.numel()], dtype=lengths.dtype, device=lengths.device)

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = radius_search(
            cur_points,
            cur_points,
            cur_lengths,
            cur_lengths,
            radius,
            neighbor_limits[i],
        )
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                radius,
                neighbor_limits[i],
            )
            subsampling_list.append(subsampling)

            upsampling = radius_search(
                cur_points,
                sub_points,
                cur_lengths,
                sub_lengths,
                radius * 2,
                neighbor_limits[i + 1],
            )
            upsampling_list.append(upsampling)

        radius *= 2

    return {
        "points": points_list,
        "lengths": lengths_list,
        "neighbors": neighbors_list,
        "subsampling": subsampling_list,
        "upsampling": upsampling_list,
    }


def single_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for single point cloud in stack mode.

    Points are organized in the following order: [P_1, ..., P_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool=True)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: feats, points, normals
    if "normals" in collated_dict:
        normals = torch.cat(collated_dict.pop("normals"), dim=0)
    else:
        normals = None
    feats = torch.cat(collated_dict.pop("feats"), dim=0)
    points_list = collated_dict.pop("points")
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    if normals is not None:
        collated_dict["normals"] = normals
    collated_dict["features"] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(
            points, lengths, num_stages, voxel_size, search_radius, neighbor_limits
        )
        collated_dict.update(input_dict)
    else:
        collated_dict["points"] = points
        collated_dict["lengths"] = lengths
    collated_dict["batch_size"] = batch_size

    return collated_dict


def registration_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)

    # Apply 'Geometric Bootstrapping' per sample
    for data_dict in data_dicts:
        ref_pts = torch.from_numpy(data_dict['ref_points']).float()
        src_pts = torch.from_numpy(data_dict['src_points']).float()

        if TEMPLATE_CACHE['r_ref'] is None:
            TEMPLATE_CACHE['r_ref'] = bufferx_radius(ref_pts)
        
        r_ref = TEMPLATE_CACHE['r_ref']
        r_src = bufferx_radius(src_pts)

        # Scale src to match ref's scale
        data_dict['ref_points'] = ref_pts 
        data_dict['src_points'] = src_pts * (r_ref / r_src)
        
        # Save metadata 
        data_dict['r_ref'] = torch.tensor(r_ref)
        data_dict['r_src'] = torch.tensor(r_src)

    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop("ref_feats") + collated_dict.pop("src_feats"), dim=0)
    points_list = collated_dict.pop("ref_points") + collated_dict.pop("src_points")
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict["features"] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(
            points, lengths, num_stages, voxel_size, search_radius, neighbor_limits
        )
        collated_dict.update(input_dict)
    else:
        collated_dict["points"] = points
        collated_dict["lengths"] = lengths
    collated_dict["batch_size"] = batch_size

    return collated_dict


def calibrate_neighbors_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True
        )

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict["neighbors"]]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits


def build_dataloader_stack_mode(
    dataset,
    collate_fn,
    num_stages,
    voxel_size,
    search_radius,
    neighbor_limits,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=False,
    distributed=False,
    precompute_data=True,
):
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            search_radius=search_radius,
            neighbor_limits=neighbor_limits,
            precompute_data=precompute_data,
        ),
        drop_last=drop_last,
        distributed=distributed,
    )
    return dataloader
