import numpy as np


def random_plane_crop(points, crop_fraction=(0.05, 0.25), rng=None):
    rng = np.random.default_rng(rng)
    axis = rng.integers(0, 3)
    frac = rng.uniform(*crop_fraction)
    lo, hi = points[:, axis].min(), points[:, axis].max()
    if rng.random() < 0.5:
        threshold = lo + frac * (hi - lo)
        mask = points[:, axis] > threshold
    else:
        threshold = hi - frac * (hi - lo)
        mask = points[:, axis] < threshold
    return points[mask]


def multi_plane_crop(points, n_cuts=(1, 4), crop_fraction=(0.03, 0.15), min_points=512, rng=None):
    rng = np.random.default_rng(rng)
    n = rng.integers(*n_cuts)
    for _ in range(n):
        result = random_plane_crop(points, crop_fraction, rng=rng)
        if len(result) < min_points:
            break
        points = result
    return points


def sphere_dropout(points, radius_fraction_range=(0.5, 1.0), rng=None):
    """Keep only points inside a sphere. Radius is a fraction of the cloud's max distance from its centroid."""
    rng = np.random.default_rng(rng)
    center = points[rng.integers(len(points))]
    max_dist = np.linalg.norm(points - points.mean(axis=0), axis=1).max()
    r = rng.uniform(*radius_fraction_range) * max_dist
    dists = np.linalg.norm(points - center, axis=1)
    return points[dists <= r]


def _has_sufficient_extent(result, original, min_fraction=0.4):
    """Reject results where any axis spans less than min_fraction of the original's span."""
    orig_span = original.max(0) - original.min(0)
    res_span = result.max(0) - result.min(0)
    valid = orig_span > 0
    if not valid.any():
        return True
    return (res_span[valid] / orig_span[valid]).min() >= min_fraction


def apply_face_augmentations(
    points,
    use_multiplane_crop=False,
    p_multiplane_crop=0.6,
    use_sphere_dropout=False,
    p_sphere_dropout=0.4,
    min_points=512,
    min_extent_fraction=0.4,
    rng=None,
):
    rng = np.random.default_rng(rng)
    original = points

    if use_multiplane_crop and rng.random() < p_multiplane_crop:
        result = multi_plane_crop(points, n_cuts=(1, 4), crop_fraction=(0.03, 0.15), min_points=min_points, rng=rng)
        if len(result) >= min_points and _has_sufficient_extent(result, original, min_extent_fraction):
            points = result

    if use_sphere_dropout and len(points) > min_points and rng.random() < p_sphere_dropout:
        result = sphere_dropout(points, radius_fraction_range=(0.5, 1.0), rng=rng)
        if len(result) >= min_points and _has_sufficient_extent(result, original, min_extent_fraction):
            points = result

    if len(points) < min_points or not _has_sufficient_extent(points, original, min_extent_fraction):
        return original
    return points
