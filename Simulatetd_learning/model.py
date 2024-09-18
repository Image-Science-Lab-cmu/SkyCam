import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import nn
import sys
from torch.autograd import Variable


class ConvMLP(nn.Module):
    def __init__(self, input_size, time_pred, selected_model):
        super(ConvMLP,self).__init__()
        self.selected_model = selected_model

        if self.selected_model == 0:
            self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=3) 
            self.relu1 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)

            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
            self.relu2 = nn.ReLU()
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)

            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3) 
            self.relu3 = nn.ReLU()
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)

            self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3) 
            self.relu4 = nn.ReLU()
            self.maxpool4 = nn.MaxPool2d(kernel_size=2)

            self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3) 
            self.relu5 = nn.ReLU()
            self.maxpool5 = nn.MaxPool2d(kernel_size=2)

            self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3) 
            self.relu6 = nn.ReLU()
            self.maxpool6 = nn.MaxPool2d(kernel_size=2)

            self.fc1 = nn.Linear(in_features=1024 * 2, out_features=1024)
            self.relu7 = nn.ReLU()

            self.fc2 = nn.Linear(in_features=1024, out_features=512)
            self.relu8 = nn.ReLU()

            self.fc3 = nn.Linear(in_features=512, out_features=256)
            self.relu9 = nn.ReLU()

            self.fc4 = nn.Linear(in_features=256, out_features=128)
            self.relu10 = nn.ReLU()

            self.fc5 = nn.Linear(in_features=128, out_features=time_pred)
            # self.sigmoind = nn.Sigmoid() # Could replace this with BCEWithLogits

        ######################################################################
        if self.selected_model == 1:
            self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=10, kernel_size=3) 
            self.relu1 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)

            self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3) 
            self.relu2 = nn.ReLU()
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)

            self.dropout = nn.Dropout(0.2)

            self.fc1 = nn.Linear(in_features= 20*48*13, out_features=2048)
            self.relu3 = nn.ReLU()

            self.fc2 = nn.Linear(in_features=2048, out_features=1024)
            self.relu4 = nn.ReLU()

            self.fc3 = nn.Linear(in_features=1024, out_features=60)
            self.relu5 = nn.ReLU()

            # self.sigmoid = nn.Sigmoid() # Could replace this with BCEWithLogits


        if self.selected_model == 2:
            self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=3) 
            self.relu1 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)

            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3) 
            self.relu2 = nn.ReLU()
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)

            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) 
            self.relu3 = nn.ReLU()
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)

            self.fc1 = nn.Linear(in_features=64*23*5, out_features=2048)
            self.relu3 = nn.ReLU()

            self.fc2 = nn.Linear(in_features=2048, out_features=1024)
            self.relu4 = nn.ReLU()

            self.fc3 = nn.Linear(in_features=1024, out_features=512)
            self.relu5 = nn.ReLU()

            self.fc4 = nn.Linear(in_features=512, out_features=60)
            self.relu6 = nn.ReLU()



    def forward(self, inputs):
        if self.selected_model == 0:
            B = inputs.shape[0]
            x = self.conv1(inputs)
            
            x = self.maxpool1(x)
            x = self.relu1(x)

            x = self.conv2(x)
            
            x = self.maxpool2(x)
            x = self.relu2(x)

            x = self.conv3(x)
            
            x = self.maxpool3(x)
            x = self.relu3(x)

            x = self.conv4(x)
            
            x = self.maxpool4(x)
            x = self.relu4(x)

            x = self.conv5(x)
        
            x = self.maxpool5(x)
            x = self.relu5(x)

            x = self.conv6(x)
            
            x = self.maxpool6(x)
            x = self.relu6(x)

            x = x.view(-1, 1024 * 2)

            x = self.fc1(x)
            x = self.relu7(x)

            x = self.fc2(x)
            x = self.relu8(x)

            x = self.fc3(x)
            x = self.relu9(x)

            x = self.fc4(x)
            x = self.relu10(x)

            x = self.fc5(x)
            # x = self.sigmoind(x)

            return(x)
        
        if self.selected_model == 1:
            x = self.conv1(inputs)
            x = self.relu1(x)
            x = self.maxpool1(x)

            x = self.conv2(x)
            x = self.relu2(x)
            x = self.maxpool2(x)
            x = self.dropout(x)


        
            x = x.view(-1,  20*48*13)

            x = self.fc1(x)
            x = self.relu3(x)
            x = self.dropout(x)

            x = self.fc2(x)
            x = self.relu4(x)

            x = self.fc3(x)

            #  Remove if doing BCE with Logits
            # x = self.relu4(x)
            # x = self.sigmoid(x)

            return(x)


        if self.selected_model == 2:
            x = self.conv1(inputs)
            x = self.relu1(x)
            x = self.maxpool1(x)

            x = self.conv2(x)
            x = self.relu2(x)
            x = self.maxpool2(x)

            x = self.conv3(x) 
            x = self.relu3(x)
            x = self.maxpool3(x)

            
            # x = x.view(-1, 64*32*23) #original
            x = x.view(-1, 64*23*5) #warp

            x = self.fc1(x)
            x = self.relu4(x)

            x = self.fc2(x)
            x = self.relu5(x)

            x = self.fc3(x)
            x = self.relu6(x)

            x = self.fc4(x)

            return(x)

        #########################


        