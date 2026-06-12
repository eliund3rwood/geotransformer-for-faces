r"""Smoke + batching-equivalence test for the shared-ref batched model.

1. Runs a B=2 batch through the model in eval mode, then the same two samples as
   two B=1 batches, and compares per-item outputs (z coefficients, estimated
   transforms) and losses. Differences should be float-noise level; coarse
   correspondence top-k near-ties may amplify tiny numeric diffs, so the check
   uses loose tolerances and reports rather than hard-fails.
2. Runs a train-mode forward+backward on B=2 and checks gradients are finite.

Run from this directory: python sanity_check.py [--ckpt PATH]
If a checkpoint is available (default: the fm_radius0.02 epoch-39 snapshot), it is
loaded so the equivalence check runs with sharp, trained features.
"""
import argparse
import os.path as osp

import numpy as np
import torch

from config import make_cfg
from dataset import SharedRefFacesDataset, shared_ref_collate_fn_stack_mode
from model import create_model
from loss import OverallLoss, Evaluator

from geotransformer.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from geotransformer.utils.data import registration_collate_fn_stack_mode, calibrate_neighbors_stack_mode
from geotransformer.utils.torch import to_cuda


DEFAULT_CKPT = osp.join(
    osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))),
    'output',
    'geotransformer.facesdownsampledfixed.stage4.gse.k3.max.oacl.stage2.sinkhorn.vl.newdata.fm_radius0.02',
    'snapshots', 'epoch-39.pth.tar',
)


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

    dataset = SharedRefFacesDataset(cfg.data.dataset_root, 'val', use_augmentation=False)
    items = [dataset[0], dataset[1]]  # fetch once so both runs see identical src points

    def collate(samples):
        return shared_ref_collate_fn_stack_mode(
            samples,
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
            neighbor_limits,
        )

    model = create_model(cfg)
    if osp.exists(args.ckpt):
        state_dict = torch.load(args.ckpt, map_location='cpu', weights_only=False)['model']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f'loaded checkpoint {args.ckpt}\n  missing: {missing}\n  unexpected: {unexpected}')
    else:
        print(f'checkpoint not found ({args.ckpt}); running with random weights')
    model.neighbor_limits = neighbor_limits
    model = model.cuda()
    loss_func = OverallLoss(cfg).cuda()
    evaluator = Evaluator(cfg).cuda()

    # --- eval-mode equivalence: B=2 vs two B=1 runs ---
    model.eval()
    with torch.no_grad():
        data_b2 = to_cuda(collate(items))
        out_b2 = model(data_b2)
        loss_b2 = loss_func(out_b2, data_b2, mode='val')
        result_b2 = evaluator(out_b2, data_b2)

        singles = []
        for i in range(2):
            data_b1 = to_cuda(collate([items[i]]))
            out_b1 = model(data_b1)
            loss_b1 = loss_func(out_b1, data_b1, mode='val')
            singles.append((out_b1, loss_b1))

    print('\n=== eval-mode equivalence (B=2 vs B=1) ===')
    ok = True
    for i in range(2):
        out_b1, _ = singles[i]
        dz = (out_b2['z_coefficients'][i] - out_b1['z_coefficients'][0]).abs().max().item()
        dt = (out_b2['estimated_transform'][i] - out_b1['estimated_transform'][0]).abs().max().item()
        dm = (out_b2['morphed_full_grad'][i] - out_b1['morphed_full_grad'][0]).abs().max().item()
        print(f'item {i}: max|dz|={dz:.3e}  max|dT|={dt:.3e}  max|d_morph|={dm:.3e}')
        if dz > 1e-2 or dm > 1e-2:
            ok = False

    mean_single = {
        k: 0.5 * (singles[0][1][k].item() + singles[1][1][k].item())
        for k in ['c_loss', 'm_loss', 'd_loss']
    }
    print('losses  B=2 batch   :', {k: round(loss_b2[k].item(), 6) for k in ['c_loss', 'f_loss', 'm_loss', 'd_loss']})
    print('losses  mean of B=1 :', {k: round(v, 6) for k, v in mean_single.items()},
          '(f_loss is a global ratio, not a per-item mean — not directly comparable)')
    print('metrics B=2 batch   :', {k: round(float(v), 4) for k, v in result_b2.items()})
    for k in ['c_loss', 'm_loss', 'd_loss']:
        if abs(loss_b2[k].item() - mean_single[k]) > 1e-2 * max(1.0, abs(mean_single[k])):
            ok = False
            print(f'WARNING: {k} differs between batched and per-item runs beyond tolerance')

    # --- train-mode smoke: forward + backward, finite grads ---
    print('\n=== train-mode smoke test (B=2, forward+backward) ===')
    model.train()
    torch.set_grad_enabled(True)
    data_b2 = to_cuda(collate(items))
    out = model(data_b2)
    loss_dict = loss_func(out, data_b2)
    print('train losses:', {k: round(v.item(), 6) for k, v in loss_dict.items()})
    loss_dict['loss'].backward()
    n_grads, n_bad = 0, 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            n_grads += 1
            if not torch.isfinite(p.grad).all():
                n_bad += 1
                print('non-finite grad:', name)
    print(f'parameters with grads: {n_grads}, non-finite: {n_bad}')
    assert n_bad == 0, 'non-finite gradients detected'
    assert torch.isfinite(loss_dict['loss']), 'non-finite loss'

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f'\npeak GPU memory: {peak:.2f} GiB')
    print('SANITY CHECK', 'PASSED' if ok else 'FINISHED WITH WARNINGS (see above)')


if __name__ == '__main__':
    main()
