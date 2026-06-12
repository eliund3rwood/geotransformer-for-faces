import os.path as osp
import pickle
import random
from functools import partial

import numpy as np
import torch
import torch.utils.data
from scipy.spatial import KDTree

from geotransformer.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    get_transform_from_rotation_translation,
)
from geotransformer.utils.data import (
    precompute_data_stack_mode,
    calibrate_neighbors_stack_mode,
    registration_collate_fn_stack_mode,
)
from geotransformer.utils.torch import build_dataloader


class SharedRefFacesDataset(torch.utils.data.Dataset):
    r"""Faces dataset for shared-reference batched training.

    Every sample registers a synthetic source scan against the SAME average-face
    template, so the reference is loaded once and kept in canonical pose. All
    augmentation (rotation, noise, subsampling, scaling, voxel-density) is applied
    to the source only. This keeps the reference frame aligned with the PCA morph
    space, and the gt transform always maps src into the canonical template frame.

    Each item returns src-side data plus the shared canonical ref.
    """

    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1.0,
        aug_scale_min=0.5,
        aug_scale_max=1.5,
        aug_subsample_keep_min=0.7,
        aug_voxel_sizes=None,
    ):
        super().__init__()

        self.dataset_root = dataset_root
        self.metadata_root = osp.join(dataset_root, 'metadata')
        self.data_root = osp.join(dataset_root, 'data')
        self.subset = subset
        self.point_limit = point_limit

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_subsample_keep_min = aug_subsample_keep_min
        self.aug_voxel_sizes = aug_voxel_sizes if aug_voxel_sizes is not None else []

        with open(osp.join(self.metadata_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)

        ref_names = set(m['pcd0'] for m in self.metadata_list)
        assert len(ref_names) == 1, f'Shared-ref dataset requires a unique ref, found {len(ref_names)}'
        ref_points = torch.load(osp.join(self.data_root, self.metadata_list[0]['pcd0']), weights_only=False)
        self.ref_points = np.asarray(ref_points, dtype=np.float32)

    def __len__(self):
        return len(self.metadata_list)

    def _load(self, file_name):
        return torch.load(osp.join(self.data_root, file_name), weights_only=False)

    def __getitem__(self, index):
        data_dict = {}

        metadata = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['ref_frame'] = metadata['frag_id0']
        data_dict['src_frame'] = metadata['frag_id1']
        data_dict['overlap'] = metadata['overlap']

        rotation = metadata['rotation']
        translation = metadata['translation']

        src_points = np.asarray(self._load(metadata['pcd1']), dtype=np.float32)
        morphed_full = np.asarray(self._load(metadata['pcd_morphed']), dtype=np.float32)
        gt_z = self._load(metadata['gt_z_path'])

        # src-only augmentation: the shared ref must stay canonical and identical
        # across the batch, so the random rotation is folded into src + transform.
        if self.use_augmentation:
            if self.aug_rotation != 0.0:  # random_sample_rotation divides by the factor
                aug_rotation = random_sample_rotation(self.aug_rotation)
                src_points = np.matmul(src_points, aug_rotation.T)
                rotation = np.matmul(rotation, aug_rotation.T)
            src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        scale_factor = 1.0
        if self.use_augmentation and self.subset != 'val':
            # Subsampling
            num_points = src_points.shape[0]
            keep_ratio = random.uniform(self.aug_subsample_keep_min, 1.0)
            keep_points = int(num_points * keep_ratio)
            indices = np.random.choice(num_points, keep_points, replace=False)
            src_points = src_points[indices]

            # Scale Augmentation (log-uniform so ×2 and ÷2 are equally likely)
            log_scale = random.uniform(np.log(self.aug_scale_min), np.log(self.aug_scale_max))
            scale_factor = np.exp(log_scale)
            src_points = src_points * scale_factor

            # Voxel-density augmentation: normalize src density to match real scanner uniformity
            if self.aug_voxel_sizes:
                vs = random.choice(self.aug_voxel_sizes)
                coords = np.floor(src_points / vs).astype(np.int32)
                keys = coords[:, 0] * 1_000_003 + coords[:, 1] * 1_009 + coords[:, 2]
                _, first_idx = np.unique(keys, return_index=True)
                if len(first_idx) >= 64:
                    src_points = src_points[first_idx]

        if self.use_augmentation and self.subset == 'val':
            # Subsampling/Upsampling
            num_points = src_points.shape[0]
            ratio = self.aug_subsample_keep_min
            target_count = int(ratio * num_points)

            if ratio > 1.0:
                num_to_interpolate = target_count - num_points

                tree = KDTree(src_points)
                indices = np.random.choice(num_points, num_to_interpolate)
                base_points = src_points[indices]
                distances, nn_indices = tree.query(base_points, k=2)
                neighbor_points = src_points[nn_indices[:, 1]]
                alphas = np.random.rand(num_to_interpolate, 1).astype(np.float32)
                interpolated_points = base_points + alphas * (neighbor_points - base_points)
                src_points = np.concatenate([src_points, interpolated_points], axis=0)
                src_points = src_points[np.random.permutation(len(src_points))]

            elif ratio < 1.0:
                indices = np.random.choice(num_points, target_count, replace=False)
                src_points = src_points[indices]

            else:
                src_points = src_points[np.random.permutation(num_points)]

            assert self.aug_scale_min == self.aug_scale_max, (
                f'aug_scale_min ({self.aug_scale_min}) != aug_scale_max ({self.aug_scale_max})'
            )
            scale_factor = self.aug_scale_min
            src_points = src_points * scale_factor

        data_dict['gt_scale'] = torch.tensor(scale_factor, dtype=torch.float32)

        transform = get_transform_from_rotation_translation(rotation, translation)

        data_dict['ref_points'] = self.ref_points
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        data_dict['morphed_full'] = morphed_full
        data_dict['gt_z'] = torch.as_tensor(gt_z, dtype=torch.float32)

        return data_dict


def shared_ref_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for shared-reference batches.

    Only the sources are stacked: points are organized as [src_1, ..., src_B] with
    per-cloud `lengths`, and the KPConv graph is precomputed on the src stack in the
    dataloader workers. The canonical ref is passed once per batch as `ref_points`;
    the model owns the (constant) ref graph. Per-sample tensors with fixed shape
    (transform, gt_z, morphed_full, gt_scale) are stacked along a batch dim.
    """
    batch_size = len(data_dicts)
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    ref_points = collated_dict.pop('ref_points')[0]
    src_points_list = collated_dict.pop('src_points')
    feats = torch.cat(collated_dict.pop('src_feats'), dim=0)
    lengths = torch.LongTensor([points.shape[0] for points in src_points_list])
    points = torch.cat(src_points_list, dim=0)

    for key in ['transform', 'gt_z', 'morphed_full', 'gt_scale']:
        if key in collated_dict:
            collated_dict[key] = torch.stack(collated_dict[key], dim=0)

    collated_dict['ref_points'] = ref_points
    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(
            points, lengths, num_stages, voxel_size, search_radius, neighbor_limits
        )
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


def train_valid_data_loader(cfg, distributed, val_aug_scale=1.0, val_aug_subsample=1.0):
    # Neighbor-limit calibration runs on the original pair dataset (joint ref+src
    # stack) so the limits cover both ref and src neighborhood statistics.
    calib_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        aug_scale_min=0.95,
        aug_scale_max=1.05,
        aug_subsample_keep_min=0.5,
        aug_voxel_sizes=[0.03, 0.04, 0.05, 0.06],
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        calib_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    train_dataset = SharedRefFacesDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        aug_scale_min=0.95,
        aug_scale_max=1.05,
        aug_subsample_keep_min=0.5,
        aug_voxel_sizes=[0.03, 0.04, 0.05, 0.06],
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        collate_fn=partial(
            shared_ref_collate_fn_stack_mode,
            num_stages=cfg.backbone.num_stages,
            voxel_size=cfg.backbone.init_voxel_size,
            search_radius=cfg.backbone.init_radius,
            neighbor_limits=neighbor_limits,
            precompute_data=True,
        ),
        drop_last=True,
        distributed=distributed,
    )

    valid_dataset = SharedRefFacesDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.test.point_limit,
        use_augmentation=True,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        aug_scale_min=val_aug_scale,
        aug_scale_max=val_aug_scale,
        aug_subsample_keep_min=val_aug_subsample,
    )
    valid_loader = build_dataloader(
        valid_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=partial(
            shared_ref_collate_fn_stack_mode,
            num_stages=cfg.backbone.num_stages,
            voxel_size=cfg.backbone.init_voxel_size,
            search_radius=cfg.backbone.init_radius,
            neighbor_limits=neighbor_limits,
            precompute_data=True,
        ),
        drop_last=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits
