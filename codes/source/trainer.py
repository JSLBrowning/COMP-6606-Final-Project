from __future__ import print_function, division

import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
from tempfile import TemporaryFile
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from skimage import transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
#import vlc
from torchsummary import summary
import cv2

# import from project functions
from copy import deepcopy


from board import GameBoard
from utils import Generate_Board, get_img_path, get_play_path, generate_dataset

# --------imports from system------
import os
import sys
from operator import itemgetter
import numpy as np
from playsound import playsound

from torch_dataload import Boardloader
from models import ConvNet1, ConvNet2, ConvNet3


trans=transforms.Compose([transforms.ToTensor()])

train_path = os.getcwd()  + '/logs/Train_Set_100000/Initial_Sates_'
label_path = os.getcwd() + '/logs/Label_Set_100000/Optimized_Data_' 
data = Boardloader(train_path,label_path,trans)

test_sample = data[0]
x = (test_sample['initial']).numpy()
y = (test_sample['labels']).numpy()

print(x.shape)
print(x)
print(y.shape)
print(y)


train_ds, valid_ds = torch.utils.data.random_split(data, (99980, 20))
dataloader = DataLoader(train_ds, batch_size=50,
                            shuffle=True, num_workers=0)
valid_dl = DataLoader(valid_ds, batch_size=10, shuffle=True)

sample_b = next(iter(dataloader))
xb = sample_b['initial']
yb = sample_b['labels']
print(xb.shape)
print(yb.shape)       


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = ConvNet3()
model = model.to(device)

# check keras-like model summary using torchsummary

summary(model, input_size=(1, 7, 7))



learning_rate = 0.001
criterion = nn.SmoothL1Loss() # Huber Loss Funtion
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
        
def train_net(n_epochs):

    # prepare the net for training
    model.train()
        
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        batch_index = 0        
        running_loss = 0.0
        summed_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(dataloader):
            # get the input images and their corresponding labels
            
            initial = data['initial']
            labels = data['labels']

            # flatten pts
            labels = labels.view(labels.size(0), -1)

            labels = labels.type(torch.cuda.FloatTensor)
            initial = initial.type(torch.cuda.FloatTensor)

            # forward pass to get outputs
            output_pts = model(initial)

            #outputs = torch.round(output_pts)
            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, labels)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            summed_loss += loss.item()
            running_loss += loss.item()
            
            if batch_i % 50 == 49:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss {}, Summed. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/50, summed_loss/50))
                running_loss = 0
    torch.save(model.state_dict(),'./3rd_Akari.pt')

train_net(50)


model.load_state_dict(torch.load('./3rd_Akari.pt'))

results_sample = next(iter(valid_dl))
inn = results_sample["initial"].type(torch.cuda.FloatTensor)
outt = results_sample["labels"].cpu().numpy()

with torch.no_grad():
    pred = model(inn)
pred = pred.reshape((10,1,7,7))
testing123 = outt[0,0,:,:]

pred123 = pred[0,0,:,:].cpu().numpy()
pred123 = np.rint(pred123)

print(testing123)
print(pred123)


pieces_dir = get_img_path()
Generate_Board(pieces_dir, testing123)
Generate_Board(pieces_dir, pred123)
