import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from dataset import VideoDataset, VideoDataset1M
from models.cls_hrnet_2dplus1 import get_cls_net
from default import _C as config

train_set = VideoDataset("/datasets/Moments_in_Time_Small", clip_len=16)
train_loader = data.DataLoader(train_set, batch_size=4, num_workers=1)
val_set = VideoDataset("/datasets/Moments_in_Time_Small", mode='val', clip_len=16)
val_loader = data.DataLoader(val_set, batch_size=4, num_workers=1)

i = 0
for dat, lab in train_loader:
    # torch.Size([3, 3, 16, 112, 112]) torch.Size([3])
    i += 4
    if i % (10*1000) == 0:
        print(i)
i = 0
for dat, lab in val_loader:
    # torch.Size([3, 3, 16, 112, 112]) torch.Size([3])
    i += 4
    if i % (10*1000) == 0:
        print(i)

print("Finished")
stop

config.defrost()
config.merge_from_file('/home/iliauk/notebooks/hr2d2/configs/cls_hrnet_w18.yaml')
device = 'cuda'
model = get_cls_net(config).to(device)

dat = dat.to(device)
out = model(dat)
stime = time.time()
for _ in range(100):
    # torch.Size([3, 339])
    #print(model(dat).size())
    out = model(dat)
# 1.4232027530670166
print(time.time() - stime)
print(model(dat).size())
