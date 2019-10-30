# -*- coding: utf-8 -*-
"""
Author: Laksh Manoj Jaisinghani
Last Modified: 20/10/19
Version: 2.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import multiprocessing
import matplotlib.pyplot as plt
import h5py
import time
import csv

from utils.model import D3
from Utile.sparse import NN_fill, generate_mask


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
CSV_FIELDNAMES = ["Epoch", "Loss"]


# 24x24 downsampling
mask = generate_mask(24, 24, 480, 640)

""" Initializing Dataset class """
class NYU_V2(Dataset):
  def __init__(self, trn_tst=0, transform=None):
    data = h5py.File('./nyu_depth_v2_labeled.mat')
    
    if trn_tst == 0:
      #trainloader
      self.images = data["images"][0:1400]
      self.depths  = data["depths"][0:1400]
    else:
      #testloader
      self.images = data["images"][1401:]
      self.depths = data["depths"][1401:]
    
    self.transform = transform
    
  def __len__(self):
    return len(self.images)
 
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
      
    sample = self.images[idx, :]
    s_depth = self.depths[idx, :]
    sample = torch.from_numpy(np.transpose(sample, (2, 1, 0)))
    s_depth=  torch.from_numpy(np.transpose(s_depth, (1, 0)))
    
    return sample.float(), s_depth.float()


""" Training funcion (per epoch) """
def train(net, device, loader, optimizer, Loss_fun):
    
  #initialise counters
  running_loss = 0.0
  loss = []
  net.train()
  torch.no_grad()
  print(1)
 
  # train batch
  start_time = time.time()
  for i, (x, y) in enumerate(loader):
    NN = []
    # concat x with spatial data
    for j in range(x.shape[0]):
      sp = NN_fill(x[j].numpy(), y[j].numpy(), mask)
      NN.append(sp)
    NN = torch.tensor(NN)
    
    optimizer.zero_grad()

    x = x.permute(0, 3, 1, 2)
    x = x.to(device) 
    y = y.to(device)
    NN = NN.to(device)
    fx = net(x, NN)
    fx = fx.permute(1, 0, 2, 3)
    loss = Loss_fun(fx[0], y)
    running_loss += loss.item()
  

    loss.backward()
    optimizer.step()

  #end training 
  end_time = time.time() 
  running_loss /= len(loader)

  print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
  torch.save(net.state_dict(), "./saved_model.pt") 
  
  return running_loss


""" Creating Train loaders """
train_set = NYU_V2(trn_tst=0) 
test_set = NYU_V2(trn_tst=1)

print(f'Number of training examples: {len(train_set)}')
print(f'Number of testing examples: {len(test_set)}')

batch_size     = 8
num_epochs     = 100
learning_rate  = 0.001
n_workers = multiprocessing.cpu_count()

#initialising data loaders
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)

img_batch, depth_batch = next(iter(trainloader))

""" Plot dataset """
plt.figure(figsize = (8, 8))

for tmp in range(4):  
    plt.subplot(1,4,tmp+1)
    plt.imshow(img_batch[tmp]/255)
    plt.title("Image")
    
plt.show()

plt.figure(figsize = (8, 8))
for tmp in range(4):  
    plt.subplot(1,4,tmp+1)
    plt.imshow(depth_batch[tmp])
    plt.title("Depth")



""" Training loop (for multiple epochs)"""
"""
Steps for skipping training:
1) check if saved_model.pt is available 
2) comment training loop
3) uncomment testing loop
"""

model = D3().float()
model = model.to(device)
Loss_fun  = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = learning_rate)


for epoch in range(num_epochs):
  
  print("Epoch", epoch + 1)
  train_loss= train(model, device, testloader, optimizer, Loss_fun)
  try:
      filepath = "./"
      with open(filepath, mode="w+") as csv_file:
          writer = csv.writer(csv_file, fieldnames=CSV_FIELDNAMES)
          writer.writerow([str(epoch+1), str(train_loss)])
  except Exception as e:
      print("log_data Error: " + str(e))


""" Testing and visualising data """
# model = D3()
# model.load_state_dict(torch.load('./saved_model.pt'))
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=n_workers)

img, depth = next(iter(testloader))

with torch.no_grad():
        model.eval()
        model.to(device)
        
        NN = []
        #concat x with spatial data
        for j in range(img.shape[0]):
          sp = NN_fill(img[j].numpy(), depth[j].numpy(), mask)
          NN.append(sp)
        NN = torch.tensor(NN)
        


        img = img.permute(0, 3, 1, 2)
        img = img.to(device) 
        depth = depth.to(device)
        NN = NN.to(device)
        print(NN.shape)
        print(img.shape)
        print(depth.shape)

        fx = model(img, NN)

""" Plot """
plt.figure(figsize = (15, 15))
tets = img[0].permute(1, 2, 0)
plt.subplot(1, 3, 1)
plt.imshow(tets/255)
plt.title("RGB")
plt.subplot(1, 3, 2)
plt.imshow(depth[0])
plt.title("Depth Ground Thruth")
plt.subplot(1, 3, 3)
plt.imshow(fx[0][0])
plt.title("Depth prediction")
