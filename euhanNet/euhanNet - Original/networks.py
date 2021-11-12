import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import save_net, load_net

class euhanNet(nn.Module):
    def __init__(self, load_weights = False):
        super(euhanNet, self).__init__() #just here
        self.seen = 0
        self.front_feat = [32,64,64,128,'M', 128,128,64] #setting up the layers (M = max-pooling)
        self.front = make_layers(self.front_feat, in_channels = 128)
        self.output_layer = nn.Conv2d(32,2, kernel_size = 1) #create output layer (convert back to 2 channel)
        self.initialize_weights()
        if load_weights: #it runs when you want pretrained network
            mod = models.vgg16(pretrained = True)
            for i in range(len(self.fron_feat.state_dict().items())):
                self.front.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]

    def forward(self, x1, x2):
        x = torch.cat((x1,x2), 1)
        x = self.front(x) #front layer
        x = self.output_layer(x) #output layer (convert back to 2 channel)
        x = F.interpolate(x, size = x1.size()[2:], mode = 'bilinear', align_corners=True)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std = 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels = 6, batch_norm = False, dilation = True):
    #set "d_rate" which defines padding and dilation
    #what is dilation?
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        elif v == 'Max':
            layers += [nn.MaxPool2d(kernel_size = 3, stride = 3)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = d_rate, dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = v
    return nn.Sequential(*layers)
