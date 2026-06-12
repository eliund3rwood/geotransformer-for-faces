r"""Throughput benchmark: batched shared-ref model (B=4) vs original model (4x B=1).

Measures train-mode forward+backward wall time per sample. The original model
builds its KPConv graph on CPU inside forward() (counted, as in real training);
the batched collate precomputes the src graph (reported separately — in real
training it runs in dataloader workers and overlaps with GPU work).

Run from this directory inside the geotransformer docker image.
"""
import argparse
import importlib.util
import os.path as osp
import time

import numpy as np
import torch

from dataset import SharedRefFacesDataset, shared_ref_collate_fn_stack_mode
from model import create_model
from loss import OverallLoss

from geotransformer.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from geotransformer.utils.data import registration_collate_fn_stack_mode, calibrate_neighbors_stack_mode
from geotransformer.utils.torch import to_cuda

ORIG_DIR = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),
                    'geotransformer.faces.stage4.gse.k3.max.oacl.stage2.sinkhorn')

WARMUP = 3
ITERS = 10


def import_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
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
        'morphed_full': torch.from_numpy(np.asarray(item['morphed_full'])),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--config', default='config', help="config module name, e.g. 'config_dowsampled'")
    parser.add_argument('--skip-original', action='store_true',
                        help='only measure the batched model (e.g. for memory probing)')
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size

    cfg = importlib.import_module(args.config).make_cfg()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    calib_dataset = ThreeDMatchPairDataset(cfg.data.dataset_root, 'train', use_augmentation=False)
    neighbor_limits = calibrate_neighbors_stack_mode(
        calib_dataset, registration_collate_fn_stack_mode,
        cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius,
    )

    dataset = SharedRefFacesDataset(cfg.data.dataset_root, 'val', use_augmentation=False)
    items = [dataset[i] for i in range(BATCH_SIZE)]

    # --- batched model, B=4 ---
    model = create_model(cfg)
    model.neighbor_limits = neighbor_limits
    model = model.cuda().train()
    loss_func = OverallLoss(cfg).cuda()

    t0 = time.perf_counter()
    data_b = to_cuda(shared_ref_collate_fn_stack_mode(
        items, cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius, neighbor_limits,
    ))
    collate_time = time.perf_counter() - t0

    for _ in range(WARMUP):
        loss_func(model(data_b), data_b)['loss'].backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        loss_func(model(data_b), data_b)['loss'].backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    batched_time = (time.perf_counter() - t0) / ITERS
    batched_mem = torch.cuda.max_memory_allocated() / 1024 ** 3

    if args.skip_original:
        print(f'\nbatched  B={BATCH_SIZE}: {batched_time * 1000:8.1f} ms/iter '
              f'({batched_time / BATCH_SIZE * 1000:6.1f} ms/sample)  peak {batched_mem:.2f} GiB '
              f'[+ collate {collate_time * 1000:.0f} ms]')
        return

    del model, loss_func, data_b
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- original model, 4x B=1 (graph built on CPU inside forward, as in training) ---
    orig_model_mod = import_from(osp.join(ORIG_DIR, 'model.py'), 'orig_model')
    orig_loss_mod = import_from(osp.join(ORIG_DIR, 'loss.py'), 'orig_loss')
    orig_model = orig_model_mod.create_model(cfg)
    orig_model.neighbor_limits = neighbor_limits
    orig_model = orig_model.cuda().train()
    orig_loss = orig_loss_mod.OverallLoss(cfg).cuda()

    data_dicts = [make_original_data_dict(item) for item in items]

    for _ in range(WARMUP):
        orig_loss(orig_model(dict(data_dicts[0])), data_dicts[0])['loss'].backward()
        orig_model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        for d in data_dicts:
            orig_loss(orig_model(dict(d)), d)['loss'].backward()
            orig_model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    orig_time = (time.perf_counter() - t0) / ITERS
    orig_mem = torch.cuda.max_memory_allocated() / 1024 ** 3

    print(f'\nbatched  B={BATCH_SIZE}: {batched_time * 1000:8.1f} ms/iter '
          f'({batched_time / BATCH_SIZE * 1000:6.1f} ms/sample)  peak {batched_mem:.2f} GiB '
          f'[+ collate {collate_time * 1000:.0f} ms, hidden in workers during training]')
    print(f'original 4x B=1: {orig_time * 1000:8.1f} ms/iter '
          f'({orig_time / BATCH_SIZE * 1000:6.1f} ms/sample)  peak {orig_mem:.2f} GiB')
    print(f'speedup per sample: {orig_time / batched_time:.2f}x')


if __name__ == '__main__':
    main()
