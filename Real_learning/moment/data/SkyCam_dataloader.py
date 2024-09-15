import h5py 
import torch 
import numpy as np 
import torch.utils.data as data
import sys 
import matplotlib.pyplot as plt

class DatasetFromFolder(data.Dataset):
    def __init__(self, input_files):
        super(DatasetFromFolder, self).__init__()

        self.inputFile = h5py.File(input_files, 'r')
        self.n_images = int(len(self.inputFile)/4)


    def __getitem__(self, index):
        XfileName = 'X' + str(index)
        YfileName = 'y' + str(index)
        target_img = 'target_img' + str(index)
        input_ghi = 'X_GHI' + str(index)
        
        # Inputs
        inputs = self.inputFile[XfileName][()]
        

        inputs = np.moveaxis(inputs, 2, 0)
        inputs = np.float32(inputs)/255
        inputs = torch.from_numpy(inputs)
        inputs = inputs.float()

        # Inputs GHI
        inputs_ghi = self.inputFile[input_ghi][()]
        inputs_ghi = torch.from_numpy(inputs_ghi)
        
        # Target GHI
        target = self.inputFile[YfileName][()]
        target = torch.from_numpy(target)

        target_img = torch.zeros((1,1), dtype=torch.float32)

        
        inputs = (inputs).float()
        inputs_ghi = (inputs_ghi).float()
        target = (target).float()


        return inputs, inputs_ghi, target, target_img

    def __len__(self):
        return int(self.n_images) #x, y, occlusion