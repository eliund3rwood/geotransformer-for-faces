r"""Equivalence test: batched shared-ref model vs the original single-pair model.

Loads the same trained checkpoint into both implementations, feeds the same val
samples (no augmentation), and compares per-item outputs.

NOTE: exact agreement is NOT expected. The library GroupNorm pools statistics
over the whole stacked point set, so the original model couples ref and src
through the norm stats; the batched backbone normalizes per cloud instead (a
requirement for batch-composition-independent outputs). This script quantifies
that drift — it should be modest (downstream outputs close), not zero.

Run from this directory (inside the geotransformer docker image):
    python compare_with_original.py [--ckpt PATH]
"""
import argparse
import importlib.util
import os.path as osp

import numpy as np
import torch

from config import make_cfg
from dataset import SharedRefFacesDataset, shared_ref_collate_fn_stack_mode
from model import create_model

from geotransformer.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from geotransformer.utils.data import registration_collate_fn_stack_mode, calibrate_neighbors_stack_mode
from geotransformer.utils.torch import to_cuda

ORIG_DIR = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),
                    'geotransformer.faces.stage4.gse.k3.max.oacl.stage2.sinkhorn')
DEFAULT_CKPT = osp.join(
    osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))),
    'output',
    'geotransformer.facesdownsampledfixed.stage4.gse.k3.max.oacl.stage2.sinkhorn.vl.newdata.fm_radius0.02',
    'snapshots', 'epoch-39.pth.tar',
)


def load_original_model_module():
    # `from backbone import KPConvFPN` inside the original model.py resolves to our
    # local (identical) backbone.py, so only model.py needs a distinct module name.
    spec = importlib.util.spec_from_file_location('orig_model', osp.join(ORIG_DIR, 'model.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_original_data_dict(item):
    ref = torch.from_numpy(np.asarray(item['ref_points']))
    src = torch.from_numpy(np.asarray(item['src_points']))
    return to_cuda({
        'points': torch.cat([ref, src], dim=0),
        'lengths': torch.LongTensor([ref.shape[0], src.shape[0]]),
        'features': torch.ones(ref.shape[0] + src.shape[0], 1),
        'transform': torch.from_numpy(np.asarray(item['transform'])),
        'gt_z': item['gt_z'],
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    args = parser.parse_args()

    cfg = make_cfg()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print('Calibrating neighbor limits...')
    calib_dataset = ThreeDMatchPairDataset(cfg.data.dataset_root, 'train', use_augmentation=False)
    neighbor_limits = calibrate_neighbors_stack_mode(
        calib_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    print('neighbor_limits:', neighbor_limits)

    state_dict = torch.load(args.ckpt, map_location='cpu', weights_only=False)['model']
    print(f'checkpoint: {args.ckpt}')

    orig_module = load_original_model_module()
    orig_model = orig_module.create_model(cfg)
    missing, unexpected = orig_model.load_state_dict(state_dict, strict=False)
    print('original  model load — missing:', missing, 'unexpected:', unexpected)
    orig_model.neighbor_limits = neighbor_limits
    orig_model = orig_model.cuda().eval()

    batched_model = create_model(cfg)
    missing, unexpected = batched_model.load_state_dict(state_dict, strict=False)
    print('batched   model load — missing:', missing, 'unexpected:', unexpected)
    batched_model.neighbor_limits = neighbor_limits
    batched_model = batched_model.cuda().eval()

    dataset = SharedRefFacesDataset(cfg.data.dataset_root, 'val', use_augmentation=False)
    items = [dataset[0], dataset[1]]

    with torch.no_grad():
        data_b2 = to_cuda(shared_ref_collate_fn_stack_mode(
            items, cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius, neighbor_limits,
        ))
        out_batched = batched_model(data_b2)

        print('\n=== batched (B=2) vs original (B=1) with trained weights ===')
        print('(diffs reflect the intended per-cloud vs pooled GroupNorm change — informational)')
        for i in range(2):
            out_orig = orig_model(make_original_data_dict(items[i]))

            d_reffeat = (out_batched['ref_feats_c'][i] - out_orig['ref_feats_c']).abs().max().item()
            d_srcfeat = (out_batched['src_feats_c'][i] - out_orig['src_feats_c']).abs().max().item()
            d_z = (out_batched['z_coefficients'][i] - out_orig['z_coefficients']).abs().max().item()
            d_morph = (out_batched['morphed_full_grad'][i] - out_orig['morphed_full_grad']).abs().max().item()
            d_T = (out_batched['estimated_transform'][i] - out_orig['estimated_transform']).abs().max().item()
            n_corr_b = out_batched['ref_corr_points'][i].shape[0]
            n_corr_o = out_orig['ref_corr_points'].shape[0]

            print(f"item {i}: max|d_ref_feats_c|={d_reffeat:.3e}  max|d_src_feats_c|={d_srcfeat:.3e}")
            print(f"         max|dz|={d_z:.3e}  max|d_morph|={d_morph:.3e}  max|dT|={d_T:.3e}")
            print(f"         fine corr count: batched={n_corr_b}  original={n_corr_o}")


if __name__ == '__main__':
    main()
