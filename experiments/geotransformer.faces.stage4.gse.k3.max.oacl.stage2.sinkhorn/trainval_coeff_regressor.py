"""
Standalone trainer for CrossAttentionRegressor — Stage 1 of a two-stage pipeline.

Stage 1 (this file): trains only CrossAttentionRegressor with morph + scale losses,
                     no backbone or matching heads involved.
Stage 2 (future):    loads Stage-1 weights into the full GeoTransformer and trains jointly.

Two-stage usage sketch:
    from trainval_coeff_regressor import CoeffRegressorTrainer, load_regressor_into_geotransformer
    from model import create_model
    from trainval_downsampled import Trainer as FullTrainer

    # --- Stage 1 ---
    stage1 = CoeffRegressorTrainer(cfg_stage1)
    stage1.run()

    # --- Stage 2 ---
    full_model = create_model(cfg_stage2)
    load_regressor_into_geotransformer(stage1.model, full_model)
    full_trainer = FullTrainer(cfg_stage2, pretrained_model=full_model)
    full_trainer.run()
"""
import copy
import time
from datetime import datetime
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch3d.ops import sample_farthest_points

from geotransformer.engine import EpochBasedTrainer
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.common import ensure_dir

from config import make_cfg
from dataset import train_valid_data_loader
from coeff_regressor import CrossAttentionRegressor, generate_reference_geometry
from viz import render_registration_3d


# ---------------------------------------------------------------------------
# Stage-1 model
# ---------------------------------------------------------------------------

class CoeffOnlyModel(nn.Module):
    """
    Lightweight Stage-1 model: CrossAttentionRegressor + PCA buffers.

    No backbone, no geometric transformer, no matching heads — forward passes
    are far cheaper than the full GeoTransformer.

    After training, call `get_regressor_state_dict()` and use
    `load_regressor_into_geotransformer()` to seed a full GeoTransformer for
    Stage-2 joint training.
    """

    def __init__(self, cfg):
        super().__init__()

        pca_data = torch.load("pca_basis_all.pth")
        self.register_buffer("pca_basis", pca_data['basis'])
        self.register_buffer("pca_mean", pca_data['mean'])
        self.register_buffer("patch_indices", pca_data['patch_indices'])

        self.coeff_regressor = CrossAttentionRegressor(
            feature_dim=cfg.model.coeff_regressor_feature_dim,
            num_patches=32,
            num_coeffs=100,
            nhead=4,
            num_layers=2,
            encoder_type=cfg.model.coeff_encoder_type,
            num_sampled_points=cfg.model.coeff_regressor_sampled_points,
            k_neighbors=cfg.model.coeff_regressor_k_neighbors,
            dropout=cfg.model.coeff_regressor_dropout,
        )

    def forward(self, data_dict):
        # Extract flat points/lengths — handles both precomputed-list and flat-tensor formats.
        if isinstance(data_dict['lengths'], list):
            flat_lengths = data_dict['lengths'][0]
            flat_points = data_dict['points'][0]
        else:
            flat_lengths = data_dict['lengths']
            flat_points = data_dict['points']

        # Collation layout: [ref_1, ..., ref_B, src_1, ..., src_B] (all refs then all srcs).
        batch_size = flat_lengths.shape[0] // 2
        num_samples = 1024
        device = flat_points.device

        ref_lengths = flat_lengths[:batch_size]
        src_lengths = flat_lengths[batch_size:]
        src_offset = ref_lengths.sum().item()

        src_raws = []       # variable-length raw src per sample (for vis)
        src_centered_list = []

        for i in range(batch_size):
            src_len = src_lengths[i].item()
            src_raw = flat_points[src_offset: src_offset + src_len]
            src_offset += src_len
            src_raws.append(src_raw)

            if src_raw.shape[0] > num_samples:
                sub, _ = sample_farthest_points(src_raw.unsqueeze(0), K=num_samples)
            elif src_raw.shape[0] < num_samples:
                # repeat points to reach num_samples so the batch can be stacked
                repeats = (num_samples + src_raw.shape[0] - 1) // src_raw.shape[0]
                sub = src_raw.repeat(repeats, 1)[:num_samples].unsqueeze(0)
            else:
                sub = src_raw.unsqueeze(0)
            src_centered_list.append(sub - sub.mean(dim=1, keepdim=True))

        # Stack into a true batch and run the regressor once.
        src_batch = torch.cat(src_centered_list, dim=0)           # [B, num_samples, 3]
        padding_mask = torch.zeros(
            (batch_size, src_batch.shape[1]), dtype=torch.bool, device=device
        )
        z_batch, scale_batch = self.coeff_regressor(src_batch, padding_mask)
        # z_batch: [B, num_patches, num_coeffs] | scale_batch: [B]

        with torch.no_grad():
            morphed_refs = [
                generate_reference_geometry(
                    z_batch[i].detach(), self.pca_basis, self.pca_mean, self.patch_indices
                )
                for i in range(batch_size)
            ]

        return {
            'z_coefficients': z_batch,    # [B, num_patches, num_coeffs]
            'pred_scale': scale_batch,    # [B]
            'morphed_refs': morphed_refs, # list of [N_verts, 3] — one per sample
            'src_points_raws': src_raws,  # list of [N_src, 3]   — one per sample
        }

    def get_regressor_state_dict(self):
        """Return only the CrossAttentionRegressor weights for weight transfer."""
        return self.coeff_regressor.state_dict()


# ---------------------------------------------------------------------------
# Stage-1 loss
# ---------------------------------------------------------------------------

def _batch_gt_field(value):
    """Normalise a gt field to a batched tensor regardless of how the collate
    function returned it (list of tensors, unbatched tensor, or already batched)."""
    if isinstance(value, list):
        return torch.stack(value)
    if isinstance(value, torch.Tensor) and value.shape == torch.Size([]):
        return value.unsqueeze(0)   # scalar → [1]
    return value


class CoeffOnlyLoss(nn.Module):
    """MSE on PCA coefficients + L1 on scale."""

    def forward(self, output_dict, data_dict):
        gt_z = _batch_gt_field(data_dict['gt_z'])
        if gt_z.dim() == 2:                    # [32, 100] → [1, 32, 100] when B=1
            gt_z = gt_z.unsqueeze(0)
        gt_scale = _batch_gt_field(data_dict['gt_scale'])
        if gt_scale.dim() == 0:                # scalar → [1] when B=1
            gt_scale = gt_scale.unsqueeze(0)

        loss_z = F.mse_loss(output_dict['z_coefficients'], gt_z)
        loss_scale = F.l1_loss(output_dict['pred_scale'], gt_scale)
        return {
            'loss': loss_z + loss_scale,
            'm_loss': loss_z,
            's_loss': loss_scale,
        }


# ---------------------------------------------------------------------------
# Weight-transfer helper (consumed by Stage-2 trainer)
# ---------------------------------------------------------------------------

def load_regressor_into_geotransformer(coeff_model, geotransformer_model):
    """
    Copy CrossAttentionRegressor weights from a trained CoeffOnlyModel into the
    coeff_regressor submodule of a full GeoTransformer for Stage-2 initialisation.
    """
    geotransformer_model.coeff_regressor.load_state_dict(
        coeff_model.get_regressor_state_dict()
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CoeffRegressorTrainer(EpochBasedTrainer):
    """
    Stage-1 trainer: optimises only CrossAttentionRegressor.

    Inherits from EpochBasedTrainer so the training loop, checkpointing,
    TensorBoard logging, and distributed setup are handled identically to the
    full Trainer in trainval_downsampled.py.
    """

    def __init__(self, cfg, output_model=None, clearml_task=None):
        self.cfg = cfg
        self.output_model = output_model
        self.clearml_task = clearml_task
        super().__init__(
            cfg,
            max_epoch=cfg.optim.max_epoch,
            save_all_snapshots=cfg.optim.save_all_snapshots,
        )

        start_time = time.time()
        train_loader, val_loader, _ = train_valid_data_loader(cfg, self.distributed)
        self.logger.info("Data loader created: {:.3f}s elapsed.".format(time.time() - start_time))
        self.register_loader(train_loader, val_loader)

        model = CoeffOnlyModel(cfg).cuda()
        model = self.register_model(model)

        # No weight decay and a higher lr — matches how the regressor is treated
        # inside the full Trainer's parameter groups.
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
        self.register_optimizer(optimizer)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay
        )
        self.register_scheduler(scheduler)

        self.loss_func = CoeffOnlyLoss().cuda()
        self.epoch_board = SummaryBoard(adaptive=True)
        self.best_mean_loss = float('inf')
        self._vis_collecting = False

    # ---- epoch hooks --------------------------------------------------------

    def before_val_epoch(self, epoch):
        # _vis_collecting is set by inference_epoch before calling super();
        # here we only reset the sample buffer for the new pass.
        self._vis_samples = []

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        if not self._vis_collecting:
            return
        if len(self._vis_samples) >= self.cfg.vis.num_samples:
            return

        gt_z = _batch_gt_field(data_dict['gt_z'])
        if gt_z.dim() == 2:
            gt_z = gt_z.unsqueeze(0)
        gt_scale = _batch_gt_field(data_dict['gt_scale'])
        if gt_scale.dim() == 0:
            gt_scale = gt_scale.unsqueeze(0)
        gt_transform = _batch_gt_field(data_dict['transform'])
        if gt_transform.dim() == 2:
            gt_transform = gt_transform.unsqueeze(0)

        batch_size = output_dict['pred_scale'].shape[0]
        for i in range(batch_size):
            if len(self._vis_samples) >= self.cfg.vis.num_samples:
                break
            self._vis_samples.append({
                'morphed_ref': output_dict['morphed_refs'][i].detach().cpu(),
                'src_points_raw': output_dict['src_points_raws'][i].detach().cpu(),
                'pred_scale': output_dict['pred_scale'][i].detach().cpu(),
                'gt_transform': gt_transform[i].detach().cpu(),
                'gt_scale': gt_scale[i].detach().cpu(),
                'pred_z': output_dict['z_coefficients'][i].detach().cpu(),
                'gt_z': gt_z[i].detach().cpu(),
            })

    def after_val_epoch(self, epoch):
        if self._vis_collecting and self._vis_samples:
            self._log_val_figures()

    def before_train_epoch(self, epoch):
        self.epoch_board.reset_all()

    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        self.epoch_board.update_from_result_dict(
            {k: v.item() if hasattr(v, 'item') else v for k, v in result_dict.items()}
        )

    def after_train_epoch(self, epoch):
        summary = self.epoch_board.summary()
        msg = '[Epoch {}] '.format(epoch) + ', '.join(
            '{}: {:.4f}'.format(k, v) for k, v in summary.items()
        )
        self.logger.critical(msg)

    def _log_val_figures(self):
        for i, s in enumerate(self._vis_samples):
            src_scaled = s['src_points_raw'] / s['pred_scale'].item()
            z_err = (s['pred_z'] - s['gt_z']).pow(2).mean().sqrt().item()
            scale_err = abs(s['pred_scale'].item() - s['gt_scale'].item())
            title = (
                f'Epoch {self.epoch} — sample {i + 1} | '
                f'pred_scale={s["pred_scale"].item():.3f}  gt_scale={s["gt_scale"].item():.3f}  '
                f'scale_err={scale_err:.3f}  z_rmse={z_err:.4f}'
            )

            if self.clearml_task:
                fig3d = render_registration_3d(
                    s['morphed_ref'],
                    src_scaled,
                    s['gt_transform'],
                    title=title,
                )
                self.clearml_task.get_logger().report_plotly(
                    title=f'val/registration/sample_{i + 1}',
                    series='3D',
                    iteration=self.epoch,
                    figure=fig3d,
                )

        self.logger.info(
            f'Validation figures logged to ClearML at epoch {self.epoch} '
            f'({len(self._vis_samples)} samples).'
        )

    # ---- step methods -------------------------------------------------------

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        return output_dict, loss_dict

    # ---- multi-scale validation ---------------------------------------------

    def write_event(self, phase, event_dict, index):
        super().write_event(phase, event_dict, index)
        if phase == 'val' and hasattr(self, '_collecting_for_scale'):
            self._scale_summaries[self._collecting_for_scale] = dict(event_dict)

    def inference_epoch(self):
        if self.epoch % 2 != 0:
            return

        original_val_loader = self.val_loader
        self._scale_summaries = {}
        do_vis = self.cfg.vis.val_freq > 0 and self.epoch % self.cfg.vis.val_freq == 0

        test_scales = [0.6, 0.8, 1.0, 1.2, 1.4]
        for scale in test_scales:
            self.logger.info(f"--- STARTING VAL PASS | Scale: {scale} ---")
            _, temp_val_loader, _ = train_valid_data_loader(
                self.cfg, self.distributed,
                val_aug_scale=scale,
                val_aug_subsample=1.0,
            )
            self.val_loader = temp_val_loader
            self._collecting_for_scale = scale
            self._vis_collecting = do_vis and (scale == 1.0)
            super().inference_epoch()
            if self._vis_collecting and self._vis_samples:
                self._log_val_figures()
            del self._collecting_for_scale
            self._vis_collecting = False

        self.val_loader = original_val_loader

        if self._scale_summaries:
            self._log_scale_metrics_figure()
            self._maybe_save_best_checkpoint()

    def _log_scale_metrics_figure(self):
        scales = sorted(self._scale_summaries.keys())
        all_metrics = sorted({k for d in self._scale_summaries.values() for k in d.keys()})

        ncols = 3
        nrows = (len(all_metrics) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, nrows * 4))
        axes = axes.flatten()

        for i, metric in enumerate(all_metrics):
            ax = axes[i]
            values = [self._scale_summaries[s].get(metric) for s in scales]
            valid = [(s, v) for s, v in zip(scales, values) if v is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, marker='o', linewidth=2)
            ax.set_title(f'Scale vs {metric}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Scale Factor', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.axvline(x=1.0, color='black', linestyle=':', alpha=0.5)

        for j in range(len(all_metrics), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        self.writer.add_figure('val/scale_metrics', fig, self.epoch)
        plt.close(fig)
        self.logger.info(f'Scale metrics figure written to TensorBoard at epoch {self.epoch}.')

    def _maybe_save_best_checkpoint(self):
        losses = [d['loss'] for d in self._scale_summaries.values() if 'loss' in d]
        if not losses:
            return
        mean_loss = sum(losses) / len(losses)
        self.writer.add_scalar('val/mean_loss_across_scales', mean_loss, self.epoch)
        if mean_loss < self.best_mean_loss:
            self.best_mean_loss = mean_loss
            self.save_snapshot('best_checkpoint.pth.tar')
            self.logger.info(
                f'New best mean loss={self.best_mean_loss:.4f} across scales at epoch {self.epoch}. '
                f'Saved best_checkpoint.pth.tar'
            )
            if self.output_model:
                checkpoint_path = osp.join(self.snapshot_dir, 'best_checkpoint.pth.tar')
                self.output_model.update_weights(checkpoint_path, target_filename='best_checkpoint.pth.tar')

    # ---- checkpoint ---------------------------------------------------------

    def checkpoint(self):
        super().checkpoint()
        if self.output_model:
            filename = f'epoch-{self.epoch}.pth.tar'
            checkpoint_path = osp.join(self.snapshot_dir, filename)
            self.output_model.update_weights(checkpoint_path, target_filename='last_checkpoint.pth.tar')
            self.logger.info(f'Checkpoint {filename} saved to ClearML OutputModel.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    from clearml import Task, OutputModel

    cfg = make_cfg()

    # Give Stage-1 its own output directory so artifacts don't collide with
    # the full joint-training run.
    cfg = copy.deepcopy(cfg)
    cfg.train.batch_size = cfg.train.coeff_regressor_batch_size
    cfg.train.num_workers = cfg.train.coeff_regressor_num_workers
    cfg.exp_name = cfg.exp_name + '.coeff_only'
    cfg.output_dir = osp.join(cfg.root_dir, 'output', cfg.exp_name)
    cfg.snapshot_dir = osp.join(cfg.output_dir, 'snapshots')
    cfg.log_dir = osp.join(cfg.output_dir, 'logs')
    cfg.event_dir = osp.join(cfg.output_dir, 'events')
    for d in [cfg.output_dir, cfg.snapshot_dir, cfg.log_dir, cfg.event_dir]:
        ensure_dir(d)

    run_name = f'{cfg.exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    task = Task.init(
        project_name='Geotransformer Faces',
        task_name=run_name,
        auto_connect_frameworks={'pytorch': False},
    )
    task.connect(dict(cfg))
    output_model = OutputModel(task=task, framework="PyTorch")

    trainer = CoeffRegressorTrainer(cfg, output_model=output_model, clearml_task=task)
    trainer.run()


if __name__ == '__main__':
    main()
