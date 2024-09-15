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

    def convert2BR(self, img, eps = 0.00000001):
        return np.expand_dims((img[:,:,2] - img[:,:,0])/((eps + img[:,:,2] + img[:,:,0])), 2)

    def __getitem__(self, index):
        XfileName = 'X' + str(index)
        YfileName = 'y' + str(index)
        target_img = 'target_img' + str(index)
        input_ghi = 'X_GHI' + str(index)
        
        # Inputs
        inputs = self.inputFile[XfileName][()]

        # Half the image
        # (B, 2001, 60, 3) -> (B, 1050, 60, 3) 
        H, W, C = inputs.shape
        # inputs = inputs[0:int(H/2)+50, :, :]
        
        inputs = np.float32(inputs)/255
        inputs = np.moveaxis(inputs, 2, 0) # (H, W, C) -> (C, H, W)
        inputs = torch.from_numpy(inputs).float()

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
        # return(100)
        return int(self.n_images) #x, y, occlusion





'''
    Combining GHI and keogram into one
'''
# import h5py 
# import torch 
# import numpy as np 
# import torch.utils.data as data
# import sys 
# import matplotlib.pyplot as plt

# class DatasetFromFolder(data.Dataset):
#     def __init__(self, input_files):
#         super(DatasetFromFolder, self).__init__()

#         self.inputFile = h5py.File(input_files, 'r')
#         self.n_images = int(len(self.inputFile)/4)

#     def convert2BR(self, img, eps = 0.00000001):
#         return np.expand_dims((img[:,:,2] - img[:,:,0])/((eps + img[:,:,2] + img[:,:,0])), 2)

#     def __getitem__(self, index):
#         XfileName = 'X' + str(index)
#         YfileName = 'y' + str(index)
#         target_img = 'target_img' + str(index)
#         input_ghi = 'X_GHI' + str(index)
        
#         # Inputs
#         inputs = self.inputFile[XfileName][()]

#         # 1 - Half the image
#         # (B, 2001, 60, 3) -> (B, 1050, 60, 3) 
#         H, W, C = inputs.shape
#         inputs_half = inputs[0:int(H/2)+50, :, :]

#         # 2 - B/R Ratio
#         # (B, 1050, 60, 3) -> (B, 1050, 60, 1) 
#         # print()
#         # print(inputs_half.shape)



#         inputs_half = np.float32(inputs_half)/255
#         # print(inputs_half.max())
#         # print(inputs_half.min())
#         inputs = self.convert2BR(inputs_half)
#         inputs = np.clip(inputs, 0, 1)

#         inputs = np.moveaxis(inputs, 2, 0) # (H, W, C) -> (C, H, W)

#         inputs = torch.from_numpy(inputs).float()

#         # Inputs GHI
#         inputs_ghi = self.inputFile[input_ghi][()]
#         inputs_ghi = torch.from_numpy(inputs_ghi)
        
#         # Target GHI
#         target = self.inputFile[YfileName][()]
#         target = torch.from_numpy(target)

#         # 3 - Combine The image with the GHI here
#         # (B, 1, 1050, 60) -> (B, 1, 1051, 60) where GHI is [:,:,0,:]
#         inputs = torch.cat((inputs_ghi.unsqueeze(0).unsqueeze(0), inputs), 1)

#         target_img = torch.zeros((1,1), dtype=torch.float32)

        
#         inputs = (inputs).float()
#         inputs_ghi = (inputs_ghi).float()
#         target = (target).float()
        


#         return inputs, inputs_ghi, target, target_img

#     def __len__(self):
#         # return(100)
#         return int(self.n_images) #x, y, occlusion