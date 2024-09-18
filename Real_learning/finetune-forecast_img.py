import numpy as np 
import h5py 
import numpy as np 
from tqdm import tqdm
from wandb import AlertLevel
import os
import subprocess
import warnings
import sys 
from copy import deepcopy
from typing import Optional

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
# from model import *
# image model
from img_model import *

import wandb
from wandb import AlertLevel


import argparse


warnings.filterwarnings("ignore")

def NAE_loss(outputs, targets, epsilon = 1e-2):    
    # percentage_loss = torch.mean(((outputs - targets) ** 2) / ((targets + epsilon) ** 2))
    percentage_loss = torch.mean(abs(outputs - targets) / (targets + epsilon))
    return percentage_loss


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_enc: torch.Tensor, input_mask: torch.Tensor = None):
        """
        Forward pass of the model.

        Parameters
        ----------
        x_enc : torch.Tensor
            Input tensor of shape (batch_size, n_channels, seq_len)
        input_mask : torch.Tensor, optional
            Input mask of shape (batch_size, seq_len), by default None

        Returns
        -------
        TimeseriesOutputs
        """
        if input_mask is None:
            batch_size, _, seq_len = x_enc.shape
            input_mask = torch.ones((batch_size, seq_len))

        if (
            self.task_name == TASKS.LONG_HORIZON_FORECASTING
            or self.task_name == TASKS.SHORT_HORIZON_FORECASTING
        ):
            return self.forecast(x_enc, input_mask)
        elif self.task_name == TASKS.IMPUTATION:
            return self.imputation(x_enc, input_mask)
        elif self.task_name == TASKS.ANOMALY_DETECTION:
            dec_out = self.anomaly_detection(x_enc, input_mask)
            return dec_out
        elif self.task_name == TASKS.CLASSIFICATION:
            return self.classification(x_enc, input_mask)
        elif self.task_name == TASKS.PRETRAINING:
            return self.pretraining(x_enc, input_mask)
        elif self.task_name == TASKS.EMBED:
            return self.embed(x_enc, input_mask)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")

    @staticmethod
    def load_pretrained_weights(
        run_name: str,
        opt_steps: Optional[int] = None,
        checkpoints_dir: str = './results/model_checkpoints',
        mirror : str = None
    ):
        path = os.path.join(checkpoints_dir, run_name)

        # if opt_steps is None:
        #     opt_steps = [int(i.split("_")[-1].split(".")[0]) for i in os.listdir(path)]
        #     opt_steps = max(opt_steps)
        #     print(f"Loading latest model checkpoint at {opt_steps} steps")

        checkpoint_path = os.path.join(path, f"MOMENT_Pretraining_checkpoint_best_{mirror}.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(path, f"MOMENT_Pretraining_checkpoint_best_{mirror}.pth")

        with open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        return checkpoint



class FineTuneForecast():
    def __init__(self, args, model, Half_img, **kwargs):
        super(FineTuneForecast, self).__init__()
        self.args = args
        self.model = model
        self.mirror = args.mirror

        self.Half_img = Half_img

        self.device = f'cuda:{self.args.device}'

        torch.cuda.set_device(self.device)

        if Half_img:
            from Real_learning.dataloader import DatasetFromFolder

        if not Half_img:
            from dataloader import DatasetFromFolder
        
        self.train_dataloader = DataLoader(DatasetFromFolder(self.args.INPUTS_PATH), batch_size=self.args.train_batch_size, shuffle=True, num_workers=1)
        self.test_dataloader =  DataLoader(DatasetFromFolder(self.args.VALID_PATH), batch_size=self.args.val_batch_size, shuffle=False, num_workers=1)

        self.forecastingSavePath = './results/forecasting/model_checkpoints' 


    def validation(self, curr_epoch, return_preds: bool = False):
        trues, preds, masks, losses = [], [], [], []
        PREDS_PER, GTS, PREDS= [], [], []

        # Getting persistence
        test_dataset = h5py.File(self.args.VALID_PATH, 'r')

        maxTestRange = int(len(test_dataset)/4) #x, y, y_image, warp+img

        if self.Half_img:
            maxTestRange = int(len(test_dataset)/4) #x, y, y_image, warp+img

        for i in tqdm(range(maxTestRange), position = 0, leave = True):
            # XfileName = 'X' + str(i)
            # inputs = test_dataset[XfileName][()]
            # inputs = np.float32(inputs)/255


            # ghi
            input_ghi_fname = 'X_GHI' + str(i)
            inputs_ghi = test_dataset[input_ghi_fname][()]



            # target
            YfileName = 'y' + str(i)
            gt = test_dataset[YfileName][()]


            persistence_pred = np.tile(inputs_ghi[-1], len(gt))

            if 0 in gt:
                continue

            PREDS_PER.append(persistence_pred)

        test_dataset.close()
        ########


        self.model.eval()
        with torch.no_grad():
            kk = 0
            plt.figure(figsize=(26, 25)) 
            for batch_x in tqdm(self.test_dataloader, total=len(self.test_dataloader)):

                inputs = Variable(batch_x[0]).float().to(self.device).squeeze(1) # input GHI associated to keogram
                inputs_ghi = Variable(batch_x[1]).float().to(self.device).unsqueeze(1) # input GHI associated to keogram
                gt = Variable(batch_x[2]).float().to(self.device).squeeze(1) # ground truth GHI
                
                # Pad the input to make it of shape 512
                padding_needed = 512 - inputs_ghi.shape[2]
                inputs_ghi = torch.nn.functional.pad(inputs_ghi, (padding_needed, 0)).to(self.device)
                B = inputs_ghi.shape[0]

                input_mask = torch.ones((B, 512)).to(self.device)
                input_mask[:, :padding_needed] = 0


                with torch.cuda.amp.autocast():
                    outputs = self.model(x_enc=inputs_ghi, input_mask=input_mask, mask=None, keogram=inputs)


                preds = outputs.forecast.squeeze(1)


                loss = self.criterion(preds, gt)


                losses.append(loss.item())

                gt_np = gt.cpu().detach().numpy()
                pred_np = preds.cpu().detach().numpy()
                # persistence_pred = np.tile(inPer, len(gt_np))


                if 0 not in gt_np:
                    # PREDS_PER.append(persistence_pred)
                    GTS.append(gt_np)
                    PREDS.append(pred_np)


                # plotting recosntruction plots here
                if kk < 10:
                    ax = plt.subplot(5,2,kk+1)

                    pred_plot = preds[0,:].cpu().detach().numpy()
                    gt_plot = gt[0,:].cpu().detach().numpy()

                    ax.plot(pred_plot, label='Pred')
                    ax.plot(gt_plot, label='gt')

                kk += 1

            losses = np.array(losses)
            average_loss = np.average(losses)


            GTS = np.vstack(GTS)
            PREDS = np.vstack(PREDS)#.squeeze(1)
            PREDS_PER = np.array(PREDS_PER)

            # Using metric from the paper
            # avg_diff = np.abs(PREDS - GTS)/GTS
            # avg_diff = np.mean(avg_diff, axis=0)
            
            # avg_diff_per = np.abs(PREDS_PER - GTS)/GTS
            # avg_diff_per = np.mean(avg_diff_per, axis=0)

            # Using new metric (sqrt l2)
            avg_diff = np.sqrt(np.mean((PREDS - GTS)**2, 0))/(np.sqrt(np.mean((GTS)**2, 0)))
            avg_diff_per = np.sqrt(np.mean((PREDS_PER - GTS)**2, 0))/(np.sqrt(np.mean((GTS)**2, 0)))


            plot_dir = os.path.join(self.args.checkpoint_path, self.run_name + '_plots')

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            
            ax.legend()
            plt.savefig(os.path.join(plot_dir, f'epoch_{curr_epoch}.png'))
            plt.close()



            ## Persistence plots
            # Add some color
            font = {'family': 'serif',
                    # 'color':  'darkred',
                    'weight': 'normal',
                    'size': 16,
                    }

            fontdict={'fontsize': plt.rcParams['axes.titlesize'],
            'fontweight': plt.rcParams['axes.titleweight'],
            }

            font = FontProperties()
            font.set_family('serif')
            font.set_style('italic')
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_title(f'MOMENT Transformer',
                        fontproperties=font)
            ax.set_xlabel('Time T+ (Minutes)', fontproperties=font)
            ax.set_ylabel(f'Normalized GHI Error $(W/m^{2})$', fontproperties=font)
            # ax.fill_between(np.arange(0, len(avg_diff_sphere_60_60)), avg_diff_para_60_60, avg_diff_sphere_60_60, where=avg_diff_para_60_60 < avg_diff_sphere_60_60, interpolate=True, color='black', alpha=0.2)
            ax.plot(avg_diff, linewidth=2., label='para', marker='*', markerfacecolor='w')
            ax.plot(avg_diff_per, linewidth=2, label='Persistence', marker='o',  linestyle='--', color='green', markerfacecolor='w')

            ax.grid(True)
            ax.set_facecolor('#e0e0e0')
            ax.set_ylim(0)
            ax.set_xlim(0)
            ax.legend(prop=font)


            custom_xtick_labels = np.arange(0.5, (60/2)+0.5, 0.5).astype(str)  # Custom tick labels
            for kk in range(0, len(custom_xtick_labels)):
                if kk % 2 == 1:
                    custom_xtick_labels[kk] = custom_xtick_labels[kk].split('.')[0]

            # Set custom ticks
            custom_xticks = np.arange(0, 60)  # Custom tick positions
            plt.xticks(custom_xticks, custom_xtick_labels)


            temp = ax.xaxis.get_ticklabels()
            n = 2
            temp = list(set(temp[:]) - set(temp[1::n]))

            for label in temp:
                label.set_visible(False)
            
            plt.savefig(os.path.join(plot_dir, f'valid_pred_plot_{curr_epoch}.png'))
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
        elif loss_type == "nae":
            criterion = NAE_loss
        return criterion
    
    def _select_optimizer(self):
        if self.args.optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.init_lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer_name == "Adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.init_lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer_name == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.init_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.args.optimizer_name} not implemented"
            )
        return optimizer
    
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


    def save_model(self,model: nn.Module,path: str,opt_steps: int,optimizer: torch.optim.Optimizer,scaler: torch.cuda.amp.GradScaler,mirror):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }

        if opt_steps is None:
            with open(os.path.join(path, f"{self.args.model_name}.pth"), "wb") as f:
                torch.save(checkpoint, f)
        else:
            if not os.path.exists(os.path.join(path, self.run_name)):
                os.makedirs(os.path.join(path, self.run_name))
            with open(
                os.path.join(
                    path, self.run_name, f"{self.args.model_name}_checkpoint_best_{mirror}.pth"
                ),
                "wb",
            ) as f:
                torch.save(checkpoint, f)


    def load_pretrained_moment(self, pretraining_task_name: str = "pre-training", do_not_copy_head: bool = True):
        pretraining_args = deepcopy(self.args)
        pretraining_args.task_name = pretraining_task_name

        checkpoint = BaseModel.load_pretrained_weights(
            run_name=pretraining_args.pretraining_run_name,
            opt_steps=pretraining_args.pretraining_opt_steps,
            mirror=self.mirror,
        )

        pretrained_model = MOMENT(configs=pretraining_args)
        pretrained_model.load_state_dict(checkpoint["model_state_dict"])

        # Copy pre-trained parameters to fine-tuned model
        for (name_p, param_p), (name_f, param_f) in zip(pretrained_model.named_parameters(), self.model.named_parameters()):
            if (name_p == name_f) and (param_p.shape == param_f.shape):
                if do_not_copy_head and name_p.startswith("head"):
                    continue
                else:
                    param_f.data = param_p.data

        self.freeze_model_parameters()  # Freeze model parameters based on fine-tuning mode

        return True
    

    def _create_results_dir(self):
        results_path = './results/forecasting'
        os.makedirs(results_path, exist_ok=True)
        return results_path

    def freeze_model_parameters(self):
        if self.args.finetuning_mode == "linear-probing":
            for name, param in self.model.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad = False
        elif self.args.finetuning_mode == "end-to-end":
            pass
        else:
            raise NotImplementedError(
                f"Finetuning mode {self.args.finetuning_mode} not implemented"
            )

        print("====== Frozen parameter status ======")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("Not frozen:", name)
            else:
                print("Frozen:", name)
        print("=====================================")

    def train(self):
        self.logger = self.setup_logger()
        self.run_name = self.logger.name

        self.checkpoint_path = os.path.join(self.args.checkpoint_path, self.run_name)
        make_dir_if_not_exists(self.checkpoint_path, verbose=True)

        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion(loss_type=self.args.loss_type, reduction="mean")

        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self._init_lr_scheduler(type=self.args.lr_scheduler_type)

        self.results_dir = self._create_results_dir()

        # Load pre-trained MOMENT model before fine-tuning
        if self.args.model_name == "MOMENT":
            self.load_pretrained_moment()

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
                gt = Variable(batch_x[2]).float().to(self.device).squeeze(1) # ground truth GHI
                
                # Pad the input to make it of shape 512
                padding_needed = 512 - inputs_ghi.shape[2]
                inputs_ghi = torch.nn.functional.pad(inputs_ghi, (padding_needed, 0)).to(self.device)
                B = inputs_ghi.shape[0]

                input_mask = torch.ones((B, 512)).to(self.device)
                input_mask[:, :padding_needed] = 0

                with torch.cuda.amp.autocast():
                    outputs = self.model(x_enc=inputs_ghi, input_mask=input_mask, mask=None, keogram=inputs)

                # We only care about the non-padded parts
                preds = outputs.forecast.squeeze(1)

                loss = self.criterion(preds, gt)

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

                # Updates the scale for next iteration.
                self.scaler.update()


                opt_steps = opt_steps + 1


                # save in checkpoint and test on validation set
                if opt_steps % self.args.checkpoint_interval == 0:
                    self.logger.alert(
                        title="Saving model",
                        text=f"Saving model after {opt_steps} steps",
                        level=AlertLevel.INFO,
                    )

                    # Only saving the best model
                    # if cur_epoch % self.args.checkpoint_interval:
                    val_loss = self.validation(cur_epoch)
                    if val_loss < lowestVal:
                        self.save_model(self.model, self.forecastingSavePath, opt_steps, self.optimizer, self.scaler,self.mirror)
                        lowestVal = val_loss
                        best_model_epoch = cur_epoch

                    print(f"    ====> : Validation loss: {val_loss:.3f}")
                    print(f"Best Model Epoch: {best_model_epoch}")
                        
                    
                
                # Adjust learning rate
                if self.args.lr_scheduler_type == "linearwarmupcosinelr":
                    self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)
                elif (
                    self.args.lr_scheduler_type == "onecyclelr"
                ):  # Should be torch schedulers in general
                    self.lr_scheduler.step()
            
            print(f"Epoch {cur_epoch}: Train loss: {loss.item():.3f}")
            cur_epoch = cur_epoch + 1
        
        return self.model






def forecast( config_path: str = "configs/forecasting/linear_probing.yaml", default_config_path: str = "configs/default.yaml", gpu_id: int = 0,
    forecast_horizon: int = 96,
    train_batch_size: int = 64,
    val_batch_size: int = 256,
    finetuning_mode: str = "linear-probing",
    init_lr: Optional[float] = None,
    max_epoch: int = 3,
    dataset_names: str = "/TimeseriesDatasets/forecasting/autoformer/electricity.csv",
    Half_img: Optional[bool] = False) -> None:
    config = Config( config_file_path=config_path, default_config_file_path=default_config_path).parse()

    control_randomness(config["random_seed"])


    config["device"] = gpu_id if torch.cuda.is_available() else "cpu"
    # config["checkpoint_path"] = PATHS.CHECKPOINTS_DIR
    args = parse_config(config)
    make_dir_if_not_exists(config["checkpoint_path"])


    model = MOMENT(args)

    print(f"Running experiments with config:\n{args}\n")
    task_obj = FineTuneForecast(args=args, model=model, Half_img=Half_img)

    NOTES = "Pre-training runs"
    task_obj.setup_logger(notes=NOTES)
    task_obj.train()
    # task_obj.end_logger()


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

    parser.add_argument(
        "--forecast_horizon", 
        type=int, 
        default=60, 
        help="Forecasting horizon"
    )

    parser.add_argument('--Half_img',
                    action="store_true",
    )

    # Extra
    parser.add_argument(
            "--train_batch_size", type=int, default=64, help="Training batch size"
        )
    parser.add_argument(
        "--val_batch_size", type=int, default=256, help="Validation batch size"
    )
    parser.add_argument(
        "--finetuning_mode", type=str, default="linear-probing", help="Fine-tuning mode"
    )  # linear-probing end-to-end-finetuning
    parser.add_argument(
        "--init_lr", type=float, default=0.00005, help="Peak learning rate"
    )
    parser.add_argument(
        "--max_epoch", type=int, default=50000, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        help="Name of dataset(s)",
        default="/TimeseriesDatasets/forecasting/autoformer/electricity.csv",
    )

        
    args = parser.parse_args()

    forecast(
        config_path=args.config,
        gpu_id=args.gpu_id,
        forecast_horizon=args.forecast_horizon,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        finetuning_mode=args.finetuning_mode,
        init_lr=args.init_lr,
        max_epoch=args.max_epoch,
        dataset_names=args.dataset_names,
        Half_img = args.Half_img
    )
