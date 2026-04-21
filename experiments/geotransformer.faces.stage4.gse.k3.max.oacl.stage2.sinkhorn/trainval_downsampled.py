import argparse
import time

import torch.optim as optim

from geotransformer.engine import EpochBasedTrainer
from geotransformer.utils.summary_board import SummaryBoard

from config_dowsampled import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator

class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch)

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
            {'params': no_decay_params, 'weight_decay': 0.0} # coeff_regressor exempted from decay
        ]
        
        optimizer = optim.Adam(optim_groups, lr=cfg.optim.lr)
        self.register_optimizer(optimizer)
        scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()
        self.epoch_board = SummaryBoard(adaptive=True)

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


    def inference_epoch(self):
        """
        Overriding the actual engine hook to perform multi-pass testing.
        """
        if self.epoch % 20 != 0:
            return

        # 1. Save original loader to restore it later
        original_val_loader = self.val_loader
        
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
        test_scales = [0.6, 0.8, 1.0, 1.2, 1.4]
        for scale in test_scales:
            self.logger.info(f"--- STARTING VAL PASS | Ratio: 1.0, Scale: {scale} ---")
            
            # Pass ratio and scale directly to the loader function
            _, temp_val_loader, _ = train_valid_data_loader(
                self.cfg, 
                self.distributed, 
                val_aug_scale=scale, 
                val_aug_subsample=1.0
            )
            
            self.val_loader = temp_val_loader
            super().inference_epoch()

        # 2. Restore the original validation loader
        self.val_loader = original_val_loader


def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":

    main()