# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)

    def forward(self, x):
    	x = x.view(x.shape[0],-1)
    	x = F.relu(self.fc1(x))
    	x = F.log_softmax(x,dim=1)
    	return x
class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 10)
    def forward(self, x):
    	x = x.view(x.shape[0],-1)
    	x = torch.tanh(self.fc1(x))
    	x = torch.tanh(self.fc2(x))
    	x = torch.tanh(self.fc3(x))
    	x = F.log_softmax(x,dim=1)
    	return x

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,50,kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(800,256)
        self.fc2 = nn.Linear(256,10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x
