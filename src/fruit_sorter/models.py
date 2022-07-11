import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I

class Net(nn.module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv_layer1 = torch.nn.Conv2d(1,32,5)
        self.pool_layer1 = torch.nn.MaxPool2d(2,2)
        self.conv_layer2 = torch.nn.Conv2d(32,64,5)
        self.pool_layer2 = torch.nn.MaxPool2d(2,2)

        self.full_connected_layer1 = torch.nn.Linear()
