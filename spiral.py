# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.fc1 = nn.Linear(2, num_hid)
        self.fc2 = nn.Linear(num_hid, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self, input):
        x = input[:,0]
        y = input[:,1]
        a = torch.atan2(y,x)
        r = torch.sqrt(torch.pow(x,2)+torch.pow(y,2))
        input =torch.stack((r,a),1)
        input = self.fc1(input)
        self.hid_1 = self.tanh(input)
        output = self.fc2(self.hid_1)
        input = self.sigmoid(output)
        return input

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.fc1 = nn.Linear(2, num_hid)
        self.fc2 = nn.Linear(num_hid, num_hid*2)
        self.fc3 = nn.Linear(num_hid*2,1)

    def forward(self, input):
        input = self.fc1(input)
        self.hid_1 = torch.tanh(input)
        self.hid_2 = self.fc2(self.hid_1)
        self.hid_2 = torch.tanh(self.hid_2)
        self.output = self.fc3(self.hid_2)
        self.output = torch.sigmoid(self.output)
        return self.output

class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()

        self.fc1 = nn.Linear(2, num_hid)
        self.fc2 = nn.Linear(num_hid+2, num_hid)
        self.fc3 = nn.Linear(num_hid+2+num_hid,1)

    def forward(self, input):
        ##input to hidden
        ini_input = self.fc1(input)
        self.hid_1 = torch.tanh(ini_input)
        #hidden layer 1
        self.hid_1_add = torch.cat((self.hid_1,input),1)
        hid_2 = self.fc2(self.hid_1_add)
        self.hid_2 = torch.tanh(hid_2)
        #Hidden layer 2
        self.hid_2_add = torch.cat((self.hid_2,input,self.hid_1),1)
        self.hid_3 = self.fc3(self.hid_2_add)
        self.output = torch.sigmoid(self.hid_3)
        return self.output

def graph_hidden(net, layer, node):
        xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
        yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
        xcoord = xrange.repeat(yrange.size()[0])
        ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
        grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)

        with torch.no_grad():  # suppress updating of gradients
            net.eval()  # toggle batch norm, dropout
            net(grid)
            if layer ==1:
                output = net.hid_1[:,node]
            else:
                output = net.hid_2[:,node]
            net.train()  # toggle batch norm, dropout back again
            pred = (output >= 0.5).float()
            # plot function computed by model
            plt.clf()
            plt.pcolormesh(xrange, yrange, pred.cpu().view(yrange.size()[0], xrange.size()[0]), cmap='Wistia')




