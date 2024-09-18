import numpy as np 
import h5py 
import numpy as np 
from tqdm import tqdm
from wandb import AlertLevel
import os
import subprocess
import warnings
import sys 

import torch
from torch.autograd import Variable
import torch.cuda.amp
from torch import optim
from torch.utils.data import DataLoader


from matplotlib.font_manager import FontProperties
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 


from moment.utils.config import Config
from moment.utils.utils import control_randomness, make_dir_if_not_exists, parse_config
from moment.common import PATHS
from momentfm.utils.masking import Masking
from moment.utils.forecasting_metrics import sMAPELoss
from moment.utils.optims import LinearWarmupCosineLRScheduler
from img_model import *
import wandb
from wandb import AlertLevel


import argparse


warnings.filterwarnings("ignore")

class Pretraining():
    def __init__(self, args, model, Half_img, mirror, **kwargs):
        super(Pretraining, self).__init__()
        self.args = args
        self.model = model
        self.mirror = mirror

        self.device = (f'cuda:{self.args.device}' if torch.cuda.is_available() else "cpu")

        # Generator to mask the inputs
        self.mask_generator = Masking(mask_ratio=0.25) # Mask 25% of patches randomly 

        torch.cuda.set_device(self.device)

        # Training Configs
        self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.init_lr,
                weight_decay=self.args.weight_decay,
            )

        if Half_img:
            from dataloader import DatasetFromFolder

        # if not Half_img:
        #     from dataloader import DatasetFromFolder
        
        self.train_dataloader = DataLoader(DatasetFromFolder(self.args.INPUTS_PATH), batch_size=self.args.train_batch_size, shuffle=True, num_workers=1)
        self.test_dataloader =  DataLoader(DatasetFromFolder(self.args.VALID_PATH), batch_size=self.args.val_batch_size, shuffle=False, num_workers=1)


    def validation(self, curr_epoch, return_preds: bool = False):
        trues, preds, masks, losses = [], [], [], []
        self.model.eval()

        with torch.no_grad():
            kk = 0
            plt.figure(figsize=(26, 25)) 
            for batch_x in tqdm(self.test_dataloader, total=len(self.test_dataloader)):

                inputs = Variable(batch_x[0]).float().to(self.device).squeeze(1) # input GHI associated to keogram
                inputs_ghi = Variable(batch_x[1]).float().to(self.device).unsqueeze(1) # input GHI associated to keogram
                gt = None #Variable(batch_x[2]).float().to(self.device).squeeze(1) # ground truth GHI
                
                # Pad the input to make it of shape 512
                padding_needed = 512 - inputs_ghi.shape[2]
                inputs_ghi = torch.nn.functional.pad(inputs_ghi+0.0000001, (padding_needed, 0)).to(self.device) #add some small value here to avoid zeros
                B = inputs_ghi.shape[0]

                input_mask = torch.ones((B, 512)).to(self.device)
                input_mask[:, :padding_needed] = 0

                # Mask the input (All 512 with a 25% random setting) (Try only randomly doing the 60)
                observation_mask = self.mask_generator.generate_mask(x=inputs_ghi, input_mask=input_mask).to(self.device).long()

                with torch.cuda.amp.autocast():
                    outputs = self.model(x_enc=inputs_ghi, input_mask=input_mask, mask=observation_mask, keogram=inputs)

                # We only care about the non-padded parts (We only care about the GHI so get the first index)
                inputs_un_padded = inputs_ghi[:,:, padding_needed:]#.unsqueeze(1)
                preds = outputs.reconstruction[:,:,padding_needed:]#.unsqueeze(1)
                out_pretrained_mask = outputs.pretrain_mask[:, padding_needed:]
                input_mask_un_padded = input_mask[:, padding_needed:]

                with torch.cuda.amp.autocast():
                    recon_loss = self.criterion(preds, inputs_un_padded)
                observed_mask = input_mask_un_padded * (1 - out_pretrained_mask)
                n_channels = preds.shape[1]
                observed_mask = observed_mask.unsqueeze(1).repeat((1, n_channels, 1))
                masked_loss = observed_mask * recon_loss
                loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

                losses.append(loss.item())

                # plotting recosntruction plots here
                if kk < 10:
                    ax = plt.subplot(5,2,kk+1)

                    pred_plot = preds[0,:,:].cpu().detach().numpy()[0]
                    gt_plot = inputs_un_padded[0,:,:].cpu().detach().numpy()[0]

                    ax.plot(pred_plot, label='Pred')
                    ax.plot(gt_plot, label='gt')

                kk += 1

            losses = np.array(losses)
            average_loss = np.average(losses)


            plot_dir = os.path.join(self.args.checkpoint_path, self.run_name + '_plots')

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            
            ax.legend()
            plt.savefig(os.path.join(plot_dir, f'epoch_{curr_epoch}.png'))
            plt.close()


            # Log in on Wandb
            self.logger.log(
                {
                    "validation_loss": average_loss
                }
            )

        self.model.train()
        return average_loss

    def setup_logger(self, notes: str = None):
        self.logger = wandb.init(
            project="Time-series Foundation Model",
            dir=PATHS.WANDB_DIR,
            config=self.args,
            name=self.args.run_name if hasattr(self.args, "run_name") else None,
            notes=self.args.notes if notes is None else notes,
            mode="disabled" if self.args.debug else "run",
        )
        if self.args.debug:
            print(f"Run name: {self.logger.name}\n")
        return self.logger


    def _select_criterion(self, loss_type: str = "mse", reduction: str = "none", **kwargs):
        if loss_type == "mse":
            criterion = nn.MSELoss(reduction=reduction)
        elif loss_type == "mae":
            criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == "huber":
            criterion = nn.HuberLoss(reduction=reduction, delta=kwargs["delta"])
        elif loss_type == "smape":
            criterion = sMAPELoss(reduction=reduction)
        return criterion
    
    
    def _init_lr_scheduler(self, type: str = "linearwarmupcosinelr"):
        decay_rate = self.args.lr_decay_rate
        warmup_start_lr = self.args.warmup_lr
        warmup_steps = self.args.warmup_steps

        if type == "linearwarmupcosinelr":
            self.lr_scheduler = LinearWarmupCosineLRScheduler(
                optimizer=self.optimizer,
                max_epoch=self.args.max_epoch,
                min_lr=self.args.min_lr,
                init_lr=self.args.init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )
        elif type == "onecyclelr":
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.args.init_lr,
                epochs=self.args.max_epoch,
                steps_per_epoch=len(self.train_dataloader),
                pct_start=self.args.pct_start,
            )
        elif type == "none":
            self.lr_scheduler = None


    def save_model(self,model: nn.Module,path: str,opt_steps: int,optimizer: torch.optim.Optimizer,scaler: torch.cuda.amp.GradScaler, mirror):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }

        if opt_steps is None:
            with open(os.path.join(path, f"{self.args.model_name}.pth"), "wb") as f:
                torch.save(checkpoint, f)
        else:
            with open(
                os.path.join(
                    path, f"{self.args.model_name}_checkpoint_best_{mirror}.pth"
                ),
                "wb",
            ) as f:
                torch.save(checkpoint, f)


    def train(self):
        self.run_name = self.logger.name
        path = os.path.join(self.args.checkpoint_path, self.run_name)
        make_dir_if_not_exists(path, verbose=True)

        self.criterion = self._select_criterion()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self._init_lr_scheduler()

        self.model.to(self.device)

        lowestVal = np.inf
        best_model_epoch = 0

        opt_steps = 0
        cur_epoch = 0
        while opt_steps < self.args.max_opt_steps or cur_epoch < self.args.max_epoch:
            self.model.train()

            for batch_x in tqdm(self.train_dataloader, total=len(self.train_dataloader)):

                self.optimizer.zero_grad(set_to_none=True)

                inputs = Variable(batch_x[0]).float().to(self.device).squeeze(1)# input GHI associated to keogram
                inputs_ghi = Variable(batch_x[1]).float().to(self.device).unsqueeze(1) # input GHI associated to keogram
                gt = None #Variable(batch_x[2]).float().to(self.device).squeeze(1) # ground truth GHI


                # Pad the input to make it of shape 512
                padding_needed = 512 - inputs_ghi.shape[2]
                inputs_ghi = torch.nn.functional.pad(inputs_ghi+0.0000001, (padding_needed, 0)).to(self.device) #add some small value here to avoid zeros
                B = inputs_ghi.shape[0]

                input_mask = torch.ones((B, 512)).to(self.device)
                input_mask[:, :padding_needed] = 0

        
                # Mask the input (All 512 with a 25% random setting) (Try only randomly doing the 60)
                observation_mask = self.mask_generator.generate_mask(x=inputs_ghi, input_mask=input_mask).to(self.device).long()

                with torch.cuda.amp.autocast():
                    outputs = self.model(x_enc=inputs_ghi, input_mask=input_mask, mask=observation_mask, keogram=inputs)


                # We only care about the non-padded parts (We only care about the GHI so get the first index)
                inputs_un_padded = inputs_ghi[:,:, padding_needed:]#.unsqueeze(1)
                preds = outputs.reconstruction[:,:,padding_needed:]#.unsqueeze(1)
                out_pretrained_mask = outputs.pretrain_mask[:, padding_needed:]
                input_mask_un_padded = input_mask[:, padding_needed:]

                with torch.cuda.amp.autocast():
                    recon_loss = self.criterion(preds, inputs_un_padded)
                observed_mask = input_mask_un_padded * (1 - out_pretrained_mask)
                n_channels = preds.shape[1]
                observed_mask = observed_mask.unsqueeze(1).repeat((1, n_channels, 1))
                masked_loss = observed_mask * recon_loss
                loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)
                

                self.logger.log(
                    {
                        "step_train_loss": loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                opt_steps = opt_steps + 1

                if opt_steps % self.args.checkpoint_interval == 0:
                    self.logger.alert(
                        title="Saving model",
                        text=f"Saving model after {opt_steps} steps",
                        level=AlertLevel.INFO,
                    )
                    if cur_epoch % self.args.checkpoint_interval:
                        val_loss = self.validation(cur_epoch)
                        if val_loss < lowestVal:
                            self.save_model(self.model, path, opt_steps, self.optimizer, self.scaler, self.mirror)
                            lowestVal = val_loss
                            best_model_epoch = cur_epoch
                        print(f"    ====> : Validation loss: {val_loss:.3f}")
                        print(f"Best Model Epoch: {best_model_epoch}")
                    
                self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)
            
            print(f"Epoch {cur_epoch}: Train loss: {loss.item():.3f}")
            cur_epoch = cur_epoch + 1
        
        return self.model




def pretrain(config_path: str = "configs/pretraining/pretrain.yaml", default_config_path: str = "configs/default.yaml", gpu_id: int = 0, Half_img: bool = False) -> None:
    config = Config( config_file_path=config_path, default_config_file_path=default_config_path).parse()

    control_randomness(config["random_seed"])


    config["device"] = gpu_id if torch.cuda.is_available() else "cpu"
    args = parse_config(config)
    make_dir_if_not_exists(config["checkpoint_path"])


    model = MOMENT(args)

    print(f"Running experiments with config:\n{args}\n")
    task_obj = Pretraining(args=args, model=model, Half_img=Half_img, mirror=config['mirror'])

    NOTES = "Pre-training runs"
    task_obj.setup_logger(notes=NOTES)
    task_obj.train()
    task_obj.end_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretraining/pretrain.yaml",
        help="Path to config file",
    )

    parser.add_argument(
        "--gpu_id", 
        type=int, 
        default=0, 
        help="GPU ID to use")
    
    parser.add_argument('--Half_img',
                    action="store_true",
    )

    args = parser.parse_args()

    pretrain(config_path=args.config, gpu_id=args.gpu_id, Half_img=args.Half_img)