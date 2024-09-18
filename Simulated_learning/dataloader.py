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
        warpImgName = 'warp_img' + str(index)
        
        inputs = self.inputFile[warpImgName]
        inputs = np.float32(inputs)/255


        inputs = np.moveaxis(inputs, 2, 0)
        inputs = torch.from_numpy(inputs)
        
        
        target = self.inputFile[YfileName][()]
        target = torch.from_numpy(target)



        target_img = self.inputFile[target_img]
        target_img = np.float32(target_img)/255
        target_img = torch.from_numpy(target_img)

        
        inputs = np.float32(inputs)
        target = np.float32(target)
        target_img = np.float32(target_img)


        return inputs, target, target_img

    def __len__(self):
        # return int(100)
        return int(self.n_images) #x, y, occlusion