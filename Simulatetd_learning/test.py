import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import ConvMLP
from dataloader import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from torch import nn
import sys 
from dataloader import DatasetFromFolder
import argparse
import h5py
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
import scipy.io as sio
from scipy.io import savemat
import os
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--mirror',
                    type=str,
                    required=True,
                    help='Mirror: hyper or sphere'
                    )
args = parser.parse_args()

mirror = args.mirror
# mirror = 'hyper'
TRAIN_WD = 200
TEST_WID = 60
NUM_DAYS = 28
NUM_SLICES = 800
INPUTS_PATH = f'./test2_day_synthetic_{mirror}_{TRAIN_WD}_{TEST_WID}_{NUM_SLICES}.h5' #Train/test is spererated by day


test_dataset = h5py.File(INPUTS_PATH, 'r')

maxTestRange = int(len(test_dataset)/4) #x, y, y_image, warp_img

# Models
devCount = torch.cuda.device_count()
dev = torch.cuda.current_device()

if devCount > 1:
    dev = "cuda:" + str(0)

device = torch.device(dev if torch.cuda.is_available() else "cpu")

torch.cuda.set_device(device)


# Select the correct model
os.makedirs(f'./saved_models/{mirror}', exist_ok=True)
final_model_path = './saved_models/{}/{}_finalModel_best.pt'.format(mirror, mirror)
pathn = final_model_path.split('/')[-1]


loader = transforms.Compose([transforms.ToTensor()])  
model = torch.load(final_model_path)
model = model.to(device)

# Make it test mode
model = model.eval()

MSE = []

PREDS = []
GTS = []


correct = torch.zeros((1, 60)).to(device)
for i in tqdm(range(maxTestRange), position = 0, leave = True):
    XfileName = 'warp_img' + str(i)
    inputs = test_dataset[XfileName][()]
    inputs = np.float32(inputs)/255


    YfileName = 'y' + str(i)
    gt = test_dataset[YfileName][()]
    gt = loader(gt)
    gt = Variable(gt).float().to(device)

    inputs = loader(inputs)
    inputs = Variable(inputs).float().to(device)
    inputs = inputs.unsqueeze(0) 


    pred = model(inputs)
    m = torch.nn.Sigmoid()
    pred = m(pred)
    pred_classes = torch.round(pred)

    # Computing Accuracy
    curr_correct = (pred_classes == gt).type(torch.uint8).squeeze(0)
    correct += curr_correct

    # Computing ROC/AUC
    gt_np = (gt.detach().cpu().numpy())[0][0]
    pred_np = (pred.detach().cpu().numpy())[0]

    GTS.append(gt_np)
    PREDS.append(pred_np)

  

GTS = np.array(GTS)
PREDS = np.array(PREDS)
AUCs_new = np.zeros((TEST_WID, 1))
plt.figure(figsize=(28, 26))  
for ii in range(TEST_WID):
    curr_truth = GTS[:, ii]
    curr_pred = PREDS[:,ii]
    fpr, tpr, _ = roc_curve(curr_truth, curr_pred )
    roc_auc = auc(fpr, tpr)

    AUCs_new[ii] = roc_auc 
    plt.subplot(10,6,ii+1)
    plt.subplots_adjust(wspace=0.5, hspace=0.5) 
    plt.title(f't+{ii+1}')
    plt.plot(fpr, tpr)
    

plt.suptitle(f'{mirror} | ROC Curve for Each Time Instant', fontsize=46)  
plt.show()





