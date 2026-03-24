import os
import torch
from geotransformer.engine import EpochBasedTrainer
from config_dowsampled import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator

class Validator(EpochBasedTrainer):
    def __init__(self, cfg, checkpoint_path, scale=1.0, subsample=1.0):
        super().__init__(cfg, max_epoch=1)

        # data loader
        _, val_loader, _ = train_valid_data_loader(cfg, self.distributed, scale, subsample)
        self.register_loader(None, val_loader)

        # model
        model = create_model(cfg).cuda()
    
        # weight loading 
        resolved_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        self.logger.info(f"Loading weights from: {resolved_path}")            
        checkpoint = torch.load(resolved_path, map_location='cuda')
    
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint.get('state_dict', checkpoint)
            
        model.load_state_dict(state_dict)
        self.register_model(model)

        # metrics
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

    def train_step(self, epoch, iteration, data_dict):
        pass

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict, epoch=epoch, iteration=iteration, mode='val')
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

def main():
    cfg = make_cfg()
    checkpoint_path = '../../output/with_aug/snapshots/epoch-40.pth.tar'
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    ratios = [1.1, 1.2, 1.3, 1.4, 1.5]
    for ratio in ratios:
        val = Validator(cfg, checkpoint_path, subsample=ratio)
        val.logger.info("Starting validation pass...")
        val.inference_epoch()

if __name__ == "__main__":
    main()