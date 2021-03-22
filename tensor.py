import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

a= torch.randn([12, 32, 30, 32, 22])
b=a.split(3,dim=2)
listt=[]
for t in b:
    listt.append(t)
t=torch.stack(listt,dim=2)
print(a.shape)
print(b[0].shape)
print(t.shape)