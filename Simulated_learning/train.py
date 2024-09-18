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
import os

parser = argparse.ArgumentParser()


parser.add_argument('--mirror',
                    type=str,
                    required=True,
                    help='Mirror: hyper or sphere'
                    )

parser.add_argument('--k_lam',
                    default=1.0,
                    type=np.float32,
                    help='(default value: %(default)s) Hyperparameter for keogram loss.')

parser.add_argument('--o_lam',
                    default=1.0,
                    type=np.float32,
                    help='(default value: %(default)s) Hyperparameter for occlusion loss.')

args = parser.parse_args()


# LR = 0.0001 #base
LR = 0.0001
BATCH_SIZE = 128
# EPOCHS = 20 # base
EPOCHS = 100
mirror = args.mirror
TRAIN_WD = 200
TEST_WID = 60
NUM_DAYS = 28
NUM_SLICES = 800
INPUTS_PATH = f'../Simulated_Data/train2_day_synthetic_{mirror}_{TRAIN_WD}_{TEST_WID}_{NUM_SLICES}.h5' #Train/test is spererated by day


# Models
devCount = torch.cuda.device_count()
dev = torch.cuda.current_device()

if devCount > 1:
    dev = "cuda:" + str(0)

device = torch.device(dev if torch.cuda.is_available() else "cpu")

# Set device for all cuda versions
torch.cuda.set_device(device)



model = ConvMLP(3, TEST_WID, 2)
model = model.to(device)



trainLoader = DataLoader(DatasetFromFolder(INPUTS_PATH), BATCH_SIZE, shuffle=True)

# Trains model (on training data) and returns the training loss
def run_train(model, x, y, target_img, BCE_loss_fn, optimizer): 
    model = model.train()

    # Pass input though model
    output = model(x)
    # print(output)
    # Loss
    loss = BCE_loss_fn(output, y)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    
    return loss.item()



# Sochastic Gradient Descent Weight Updater
optimizer = optim.Adam(model.parameters(), lr = LR)


# Training Loss
train_loss = []

# Losses
# BCE_loss_fn = nn.BCELoss().to(device)
BCE_loss_fn = nn.BCEWithLogitsLoss().to(device)

# Training Part ...
num_images = 0
for epoch in tqdm(range(EPOCHS), position = 0, leave = True):
    print('Starting Epoch...', epoch + 1)
    
    trainLossCount = 0
    num_images = 0
    for i, data in enumerate(trainLoader):
        # Training
        inputs = Variable(data[0]).to(device) # The input keoogram
        target = Variable(data[1]).float().to(device).squeeze(1) # The binary GT occlusion states
        target_img = Variable(data[2]).float().to(device) # The GT slice

        
        num_images += inputs.size(0)
        
        # Trains model
        trainingLoss = run_train(model, inputs, target, target_img, BCE_loss_fn, optimizer) # Add skip as a parameter here
        trainLossCount = trainLossCount + trainingLoss
        
    
    epoch_loss = trainLossCount#/num_images
    train_loss.append(epoch_loss)

    print('Training Loss...')
    print("===> Epoch[{}]({}/{}): Loss: {:.8f}".format(epoch + 1, i + 1, len(trainLoader), epoch_loss))

    if epoch % 10 == 0:
        os.makedirs(f'./saved_models/{mirror}/model_iterations/', exist_ok=True)
        PATH =  './saved_models/{}/model_iterations/{}_Iteration'.format(mirror, mirror) + str(epoch) + '.pt'
        torch.save(model, PATH)
        print('Saved model iteration ' +  str(epoch) + ' to -> ' + PATH)


print('Training Complete...')

os.makedirs(f'./saved_models/{mirror}', exist_ok=True)
PATH =  './saved_models/{}/{}_finalModel.pt'.format(mirror, mirror)
torch.save(model, PATH)
print('Saved Final Model to -> ' + PATH)