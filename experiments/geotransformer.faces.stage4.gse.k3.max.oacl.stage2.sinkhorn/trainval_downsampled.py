import argparse
import time
from datetime import datetime

import matplotlib
from clearml import Task, OutputModel
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from geotransformer.engine import EpochBasedTrainer
from geotransformer.utils.summary_board import SummaryBoard

from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator
from viz import render_registration_figure
import os.path as osp

class Trainer(EpochBasedTrainer):
    def __init__(self, cfg, output_model=None):
        self.cfg = cfg
        self.output_model = output_model
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch, save_all_snapshots=cfg.optim.save_all_snapshots)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = "Data loader created: {:.3f}s collapsed.".format(loading_time)
        self.logger.info(message)
        message = "Calibrate neighbors: {}.".format(neighbor_limits)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model, optimizer, scheduler
        model = create_model(cfg).cuda()
        model.neighbor_limits = neighbor_limits
        model = self.register_model(model)
    
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'coeff_regressor' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
        optim_groups = [
            {'params': decay_params, 'weight_decay': cfg.optim.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0, 'lr': 1e-3} # coeff_regressor exempted from decay
        ]
        
        optimizer = optim.Adam(optim_groups, lr=cfg.optim.lr)
        self.register_optimizer(optimizer)
        scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()
        self.epoch_board = SummaryBoard(adaptive=True)

        self._load_debug_samples()
        self.best_mean_loss = float('inf')


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

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)   

        loss_dict = self.loss_func(output_dict, data_dict, epoch, iteration, mode='train')
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)

        
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict, epoch=epoch, iteration=iteration, mode='val')
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        if not getattr(self, '_vis_collecting', False):
            return
        if len(self._vis_samples) >= self.cfg.vis.num_samples:
            return
        self._vis_samples.append({
            'ref_pts': output_dict['ref_points'].detach().cpu(),
            'src_pts': output_dict['src_points'].detach().cpu(),
            'estimated_T': output_dict['estimated_transform'].detach().cpu(),
        })

    def _log_registration_figures(self, tag_prefix='val/registration'):
        for i, sample in enumerate(self._vis_samples):
            fig = render_registration_figure(
                sample['ref_pts'],
                sample['src_pts'],
                sample['estimated_T'],
                title=f'Epoch {self.epoch} — sample {i + 1}',
            )
            self.writer.add_figure(f'{tag_prefix}/sample_{i + 1}', fig, self.epoch)
            plt.close(fig)
        self.logger.info(
            f'Registration figures written to TensorBoard at epoch {self.epoch} '
            f'({len(self._vis_samples)} samples).'
        )

    def write_event(self, phase, event_dict, index):
        """Intercept val events during multi-scale loop to cache per-scale summaries."""
        super().write_event(phase, event_dict, index)
        if phase == 'val' and hasattr(self, '_collecting_for_scale'):
            self._scale_summaries[self._collecting_for_scale] = dict(event_dict)

    def inference_epoch(self):
        """
        Overriding the actual engine hook to perform multi-pass testing.
        """
        if self.epoch % 2 != 0:
            return

        original_val_loader = self.val_loader
        self._scale_summaries = {}

        # --- Pass 1: Varying Ratios (Subsample) | Fixed Scale at 1.0 ---
        # test_ratios = [0.5, 0.75, 1.0]
        # for ratio in test_ratios:
        #     self.logger.info(f"--- STARTING VAL PASS | Ratio: {ratio}, Scale: 1.0 ---")

        #     # Pass ratio and scale directly to the loader function
        #     _, temp_val_loader, _ = train_valid_data_loader(
        #         self.cfg,
        #         self.distributed,
        #         val_aug_scale=1.0,
        #         val_aug_subsample=ratio
        #     )

        #     self.val_loader = temp_val_loader
        #     super().inference_epoch()

        # --- Pass 2: Varying Scales | Fixed Ratio at 1.0 ---
        do_vis = (
            self.cfg.vis.val_freq > 0
            and self.epoch % self.cfg.vis.val_freq == 0
        )
        test_scales = [0.6, 0.8, 1.0, 1.2, 1.4]
        for scale in test_scales:
            self.logger.info(f"--- STARTING VAL PASS | Ratio: 1.0, Scale: {scale} ---")
            _, temp_val_loader, _ = train_valid_data_loader(
                self.cfg,
                self.distributed,
                val_aug_scale=scale,
                val_aug_subsample=1.0
            )
            self.val_loader = temp_val_loader
            self._collecting_for_scale = scale
            # Collect registration samples only on the identity-scale pass
            self._vis_collecting = do_vis and (scale == 1.0)
            self._vis_samples = []
            super().inference_epoch()
            if self._vis_collecting and self._vis_samples:
                self._log_registration_figures(tag_prefix='val/registration')
                self._log_debug_figures()
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
        losses = [
            d['loss'] for d in self._scale_summaries.values()
            if 'loss' in d
        ]
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

    def _load_debug_samples(self):
        exp_dir = osp.dirname(osp.realpath(__file__))
        ref_np = np.load(osp.join(exp_dir, 'ref.npy')).astype(np.float32)
        pca_data = torch.load(osp.join(exp_dir, 'pca_basis_all.pth'), map_location='cpu')
        gt_z_np = np.zeros((pca_data['patch_indices'].shape[0], 100), dtype=np.float32)
        self._debug_samples = []
        for name in ['plank', '2189']:
            path = osp.join(exp_dir, f'{name}.npy')
            if osp.exists(path):
                self._debug_samples.append((name, ref_np, np.load(path).astype(np.float32), gt_z_np))
                self.logger.info(f'Loaded debug sample: {name}.npy')

    def _log_debug_figures(self):
        if not self._debug_samples:
            return
        device = next(self.model.parameters()).device
        for name, ref_np, src_np, gt_z_np in self._debug_samples:
            data_dict = _build_debug_data_dict(ref_np, src_np, gt_z_np, device)
            with torch.no_grad():
                output_dict = self.model(data_dict)
            pred_scale = output_dict['pred_scale'].item()
            fig = render_registration_figure(
                output_dict['ref_points'],
                output_dict['src_points'],
                output_dict['estimated_transform'],
                title=f'Epoch {self.epoch} — {name} (pred_scale={pred_scale:.3f})',
            )
            self.writer.add_figure(f'val/debug_registration/{name}', fig, self.epoch)
            plt.close(fig)
        self.logger.info(f'Debug registration figures written to TensorBoard at epoch {self.epoch}.')

    def checkpoint(self):
        super().checkpoint()
        if self.output_model:
            filename = f'epoch-{self.epoch}.pth.tar'
            checkpoint_path = osp.join(self.snapshot_dir, filename)
            self.output_model.update_weights(checkpoint_path, target_filename='last_checkpoint.pth.tar')
            self.logger.info(f'Model checkpoint {filename} saved to ClearML OutputModel.')


def _build_debug_data_dict(ref_pts_np, src_pts_np, gt_z_np, device):
    ref_t = torch.from_numpy(ref_pts_np).to(device)
    src_t = torch.from_numpy(src_pts_np).to(device)
    points = torch.cat([ref_t, src_t], dim=0)
    lengths = torch.tensor([ref_t.shape[0], src_t.shape[0]], dtype=torch.long, device=device)
    features = torch.ones((points.shape[0], 1), dtype=torch.float32, device=device)
    return {
        'points': points,
        'lengths': lengths,
        'features': features,
        'transform': torch.eye(4, dtype=torch.float32, device=device),
        'gt_z': torch.from_numpy(gt_z_np).to(device),
        'batch_size': 1,
    }


def main():
    cfg = make_cfg()
    run_name = f'{cfg.exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    task = Task.init(project_name='Geotransformer Faces', task_name=run_name,  auto_connect_frameworks={'pytorch': False}  )
    task.connect(dict(cfg))
    output_model = OutputModel(task=task, framework="PyTorch")
    trainer = Trainer(cfg, output_model=output_model)
    trainer.run()


if __name__ == "__main__":

    main()