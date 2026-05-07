import numpy as np
import torch.utils.data

from geotransformer.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)
from augmentations import apply_face_augmentations


class FaceAugDataset(torch.utils.data.Dataset):
    """Wraps ThreeDMatchPairDataset and applies face-specific augmentations
    (multi-plane crop, sphere dropout) independently to ref and src point clouds."""

    def __init__(self, base_dataset, use_multiplane_crop, p_multiplane_crop, use_sphere_dropout, p_sphere_dropout, min_extent_fraction=0.4):
        self.base = base_dataset
        self.use_multiplane_crop = use_multiplane_crop
        self.p_multiplane_crop = p_multiplane_crop
        self.use_sphere_dropout = use_sphere_dropout
        self.p_sphere_dropout = p_sphere_dropout
        self.min_extent_fraction = min_extent_fraction

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        data_dict = self.base[index]
        kwargs = dict(
            use_multiplane_crop=self.use_multiplane_crop,
            p_multiplane_crop=self.p_multiplane_crop,
            use_sphere_dropout=self.use_sphere_dropout,
            p_sphere_dropout=self.p_sphere_dropout,
            min_extent_fraction=self.min_extent_fraction,
        )
        data_dict['src_points'] = apply_face_augmentations(data_dict['src_points'], **kwargs)
        data_dict['ref_feats'] = np.ones((len(data_dict['ref_points']), 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((len(data_dict['src_points']), 1), dtype=np.float32)
        return data_dict


def train_valid_data_loader(cfg, distributed, val_aug_scale=1.0, val_aug_subsample=1.0):
    train_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        # ======================================
        aug_scale_min=0.5,
        aug_scale_max=1.5,
        aug_subsample_keep_min=0.5,
        # ======================================
    )
    use_face_aug = cfg.train.use_sphere_dropout or cfg.train.use_multiplane_crop
    if use_face_aug:
        train_dataset = FaceAugDataset(
            train_dataset,
            use_multiplane_crop=cfg.train.use_multiplane_crop,
            p_multiplane_crop=cfg.train.p_multiplane_crop,
            use_sphere_dropout=cfg.train.use_sphere_dropout,
            p_sphere_dropout=cfg.train.p_sphere_dropout,
            min_extent_fraction=cfg.train.aug_min_extent_fraction,
        )

    if cfg.train.max_samples is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(cfg.train.max_samples))

    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
        precompute_data=False,
    )

    valid_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.test.point_limit,
        #use_augmentation=False,
        use_augmentation=True,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        # ======================================
        aug_scale_min= val_aug_scale,
        aug_scale_max= val_aug_scale,
        aug_subsample_keep_min= val_aug_subsample,
        # ======================================
    )

    if cfg.test.max_samples is not None:
        valid_dataset = torch.utils.data.Subset(valid_dataset, range(cfg.test.max_samples))

    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        precompute_data=False,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg, benchmark):
    train_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
     
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, neighbor_limits