"""
Author: Zhuo Su, Wenzhe Liu
Date: Feb 18, 2021
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def make_layers(cfg, in_channels = 3, batch_norm=False, dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class block(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
    def __init__(self, in_channels):
        super(block, self).__init__()
        

        self.conv0_0 = conv_layers(in_channels, in_channels,1)
        self.conv0_1 = conv_layers(in_channels, in_channels,2)

        self.pool0 = pool_layers()
        self.poollast = pool_layers()

        self.conv1_0 = conv_layers(in_channels, in_channels,2)
        self.conv1_1 = conv_layers(in_channels, in_channels,2)

        self.pool1 = pool_layers()

        self.conv2_0 = conv_layers(in_channels, in_channels,2)
        self.conv2_1 = conv_layers(in_channels, in_channels,2)
        self.conv2_2 = conv_layers(in_channels, in_channels,2)

        self.pool2 = pool_layers()

        self.conv3_0 = conv_layers(in_channels, in_channels,2)
        self.conv3_1 = conv_layers(in_channels, in_channels,2)
        self.conv3_2 = conv_layers(in_channels, in_channels,2)
        self.classifier = nn.Conv2d(in_channels*3, 1, kernel_size=1)
        
    def forward(self, x):
        H, W = x.size()[2:]

        x = self.conv0_0(x)
        x = self.conv0_1(x)
  

        x = self.pool0(x)

        x = self.conv1_0(x)
      
        x = self.conv1_1(x)
        e1 = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        
        x = self.pool1(x)
      
        x = self.conv2_0(x)
       
        x = self.conv2_1(x)
        
        x = self.conv2_2(x)
        e2 = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        
        x = self.pool2(x)

        x = self.conv3_0(x)
        
        x = self.conv3_1(x)
        
        x = self.conv3_2(x)
        e3 = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        
        return e1+e2+e3
        
class blockbn(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
    def __init__(self, in_channels):
        super(blockbn, self).__init__()
        #self.norm_layer = get_norm_layer(norm_type='batch')
        norm_layer = get_norm_layer(norm_type='batch')
        

        self.conv0_0 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))
        self.conv0_1 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))

        self.pool0 = pool_layers()
        

        self.conv1_0 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))
        self.conv1_1 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))

        self.pool1 = pool_layers()

        self.conv2_0 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))
        self.conv2_1 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))
        self.conv2_2 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))

        self.pool2 = pool_layers()

        self.conv3_0 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))
        self.conv3_1 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))
        self.conv3_2 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))
        self.classifier = nn.Conv2d(in_channels*3, 1, kernel_size=1)
    def _conv_block(self, in_nc, out_nc, norm_layer, num_block=1, kernel_size=3, 
        stride=1, padding=1, bias=False):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride, 
                               padding=padding, bias=bias),
                     norm_layer(out_nc),
                     nn.ReLU(True)]
        return conv
        
    def forward(self, x):
        H, W = x.size()[2:]

        x = self.conv0_0(x)
        x = self.conv0_1(x)
       

        x = self.pool0(x)

        x = self.conv1_0(x)
        
        x = self.conv1_1(x)
        e1 = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        
        x = self.pool1(x)
      
        x = self.conv2_0(x)
        
        x = self.conv2_1(x)
        
        x = self.conv2_2(x)
        e2 = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        
        x = self.pool2(x)

        x = self.conv3_0(x)
        
        x = self.conv3_1(x)
        
        x = self.conv3_2(x)
        e3 = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        return e1+e2+e3
        
def conv_layers(inp, oup, dilation):
    #if dilation:
    d_rate = dilation
    #else:
    #    d_rate = 1
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, padding=d_rate, dilation=d_rate),
        nn.ReLU(inplace=True)
    )


def feature_transform(inp, oup):
    conv2d = nn.Conv2d(inp, oup, kernel_size=1)  # no padding
    relu = nn.ReLU(inplace=True)
    layers = []
    layers += [conv2d, relu]
    return nn.Sequential(*layers)


def pool_layers(ceil_mode=True):
    return nn.MaxPool2d(kernel_size=3, stride=2)


class CHRNet(nn.Module): # one conv layer after fully connetced CNN _average3
    def __init__(self):
        super(CHRNet, self).__init__()
        self.seen = 0


        self.conv0_0 = conv_layers(3, 16,4)
        self.pool00 = pool_layers()
        self.conv0_1 = conv_layers(16, 16,4)

        

        self.pool0 = block(16)
        self.pool0_bn = blockbn(16)

        self.conv1_0 = conv_layers(16, 32,4)
        self.conv1_1 = conv_layers(32, 32,4)

        self.pool1 = block(32)
        self.pool1_bn = blockbn(32)
        
        self.conv2_0 = conv_layers(32, 64,4)
        self.conv2_1 = conv_layers(64, 64,4)
        self.conv2_2 = conv_layers(64, 64,4)

        self.pool2 = block(64)        
        self.pool2_bn = blockbn(64)
        
        self.conv3_0 = conv_layers(64, 128,4)
        self.conv3_1 = conv_layers(128, 128,4)
        self.conv3_2 = conv_layers(128,128,4)

        self.output_layer0 = nn.Conv2d(16, 1, kernel_size=1)
        self.output_layer1 = nn.Conv2d(32, 1, kernel_size=1)
        self.output_layer2 = nn.Conv2d(64, 1, kernel_size=1)
        self.output_layer3 = nn.Conv2d(256, 1, kernel_size=1)

        self.output_layer4 = nn.Conv2d(128, 1, kernel_size=1)
        self.classifier1 = nn.Conv2d(2, 1, kernel_size=1)
        self.classifier4 = nn.Conv2d(4, 1, kernel_size=1)
        self.classifier= nn.Conv2d(6, 1, kernel_size=1)



        #self._initialize_weights()
        self.features = []
    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)

        return conv_weights, bn_weights, relu_weights
    def forward(self, x):
        self.features = []
        H, W = x.size()[2:]

        x = self.conv0_0(x)
        x = self.pool00(x)
        x = self.conv0_1(x)

        x1 = self.pool0(x)
        

        x = self.conv1_0(x+x1)
        y1 = self.conv1_1(x)
        
        e11 = F.interpolate(y1, (H, W), mode="bilinear")
        
        x2_bn = self.pool1_bn(y1)

        e22 = F.interpolate(x2_bn, (H, W), mode="bilinear")


        x = self.conv2_0(x2_bn+y1)

        x = self.conv2_1(x)
        y2 = self.conv2_2(x)

        x3_bn = self.pool2_bn(y2)


        e33 = F.interpolate(x3_bn, (H, W), mode="bilinear")
        x = self.conv3_0(x3_bn+y2)
        
        y3 = self.conv3_1(x)
        e3 = F.interpolate(y3, (H, W), mode="bilinear")
        

        e1 = F.interpolate(y1, (H, W), mode="bilinear")
        e2 = F.interpolate(y2, (H, W), mode="bilinear")
        
        e1=self.output_layer1(e1)
        e11=self.output_layer1(e11)
        e2=self.output_layer2(e2)
        e22=self.output_layer1(e22)
        e3= self.output_layer4(e3)
        e33=self.output_layer2(e33)


        

        outputs = [e1,e11,e2,e22,e3,e33]

        x = self.classifier(torch.cat(outputs, dim=1))

        outputs.append(x)


        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs






def chrnet(args):

    return CHRNet()


