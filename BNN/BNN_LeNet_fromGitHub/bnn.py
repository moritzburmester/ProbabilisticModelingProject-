import math
import torch.nn as nn
from bnn_layers.BNN_Conv2d import BNN_Conv2d
from bnn_layers.BNN_Linear import BNN_Linear
from misc import FlattenLayer, ModuleWrapper

class BNNLeNet(ModuleWrapper):

    def __init__(self, outputs, inputs, priors, activation_type='softplus'):
        super(BNNLeNet, self).__init__()

        self.num_classes = outputs
        self.priors = priors
 
        BNNLinear = BNN_Linear
        BNNConv2d = BNN_Conv2d
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BNNConv2d(inputs, 6, 5, padding=0, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BNNConv2d(6, 16, 5, padding=0, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = BNNLinear(5 * 5 * 16, 120, bias=True, priors=self.priors)
        self.act3 = self.act()

        self.fc2 = BNNLinear(120, 84, bias=True, priors=self.priors)
        self.act4 = self.act()

        self.fc3 = BNNLinear(84, outputs, bias=True, priors=self.priors)

