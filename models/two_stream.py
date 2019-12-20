from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

# Batchnorm
BatchNorm2d = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=bias)

def conv3x1x1(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=[3,1,1], 
                    stride=[stride,1,1], padding=[padding,0,0], bias=bias)

def conv1x3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=[1,3,3], 
                    stride=[1, stride, stride], padding=[0, padding, padding], bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class TwoStream(nn.Module):
    def __init__(self, layers, num_classes=39):
        super(TwoStream, self).__init__()

        self.inplanes = 64
        self.temporal_expansion = 16
        # Input is (B, 3, 32, 128, 128)
        # STEM follows HRNet (conv2 no downsample)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=3, 
            stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv3d(self.inplanes, self.inplanes, kernel_size=3, 
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(self.inplanes, momentum=BN_MOMENTUM)

        # Output is (B, 64, 16, 64, 64)

        # Split Tensor for different branches
        # TemporalChannel -> Pool Channels to 1
        self.temp_conv1 = nn.Conv3d(64, 1, kernel_size=(1,1,1), padding=0)
        # SpatialChannel -> Pool Temporal to 1
        self.spat_conv1 = nn.Conv3d(16, 1, kernel_size=(1,1,1), padding=0)
        
        # Layer1
        _layer1_channels = 64
        # Spatial
        self.spat_layer1 = self._make_spat_layer(BasicBlock, _layer1_channels, layers[0])
        # Temporal
        self.temp_layer1_conv1 = conv3x1x1(1, self.temporal_expansion)
        self.temp_layer1_bn1 = nn.BatchNorm3d(self.temporal_expansion)
        self.temp_layer1_conv2 = conv3x1x1(self.temporal_expansion, 1)
        self.temp_layer1_bn2 = nn.BatchNorm3d(1)
        # Fusion
        self.temp_to_spat_layer1 = conv1x1(16, _layer1_channels)
        self.spat_to_temp_layer1 = conv1x1(_layer1_channels, 16)

        # Layer2
        _layer2_channels = 128
        # Spatial
        self.spat_layer2 = self._make_spat_layer(BasicBlock, _layer2_channels, layers[1], stride=2)
        # Temporal
        self.temp_layer2_conv1 = conv1x3x3(1, self.temporal_expansion, stride=2)  # Reduce spatially
        self.temp_layer2_bn1 = nn.BatchNorm3d(self.temporal_expansion)
        self.temp_layer2_conv2 = conv3x1x1(self.temporal_expansion, 1)
        self.temp_layer2_bn2 = nn.BatchNorm3d(1)
        # Fusion
        self.temp_to_spat_layer2 = conv1x1(16, _layer2_channels)
        self.spat_to_temp_layer2 = conv1x1(_layer2_channels, 16)

        # Layer3
        _layer3_channels = 256
        # Spatial
        self.spat_layer3 = self._make_spat_layer(BasicBlock, _layer3_channels, layers[1], stride=2)
        # Temporal
        self.temp_layer3_conv1 = conv1x3x3(1, self.temporal_expansion, stride=2)  # Reduce spatially
        self.temp_layer3_bn1 = nn.BatchNorm3d(self.temporal_expansion)
        self.temp_layer3_conv2 = conv3x1x1(self.temporal_expansion, 1)
        self.temp_layer3_bn2 = nn.BatchNorm3d(1)
        # Fusion
        self.temp_to_spat_layer3 = conv1x1(16, _layer3_channels)
        self.spat_to_temp_layer3 = conv1x1(_layer3_channels, 16)

        # Layer4
        _layer4_channels = 512
        # Spatial
        self.spat_layer4 = self._make_spat_layer(BasicBlock, _layer4_channels, layers[1], stride=2)
        # Temporal
        self.temp_layer4_conv1 = conv1x3x3(1, self.temporal_expansion, stride=2)  # Reduce spatially
        self.temp_layer4_bn1 = nn.BatchNorm3d(self.temporal_expansion)
        self.temp_layer4_conv2 = conv3x1x1(self.temporal_expansion, 1)
        self.temp_layer4_bn2 = nn.BatchNorm3d(1)
        # Fusion
        self.temp_to_spat_layer4 = conv1x1(16, _layer4_channels)
        self.spat_to_temp_layer4 = conv1x1(_layer4_channels, 16)

        # Head
        self.head_temp_conv1 = conv1x1(16, _layer4_channels)
        self.head_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.head_fc = nn.Linear(512, num_classes)


    def _make_spat_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
            # STEM
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            # Create tensors for two-streams
            # torch.Size([2, 64, 64, 64])
            spat_x = torch.squeeze(self.spat_conv1(x.permute(0,2,1,3,4)))
            # torch.Size([2, 1, 16, 64, 64])
            temp_x = self.temp_conv1(x)

            # Layer1
            spat_x = self.spat_layer1(spat_x)
            #temp_x.unsqueeze_(1)
            temp_x = self.temp_layer1_conv1(temp_x)
            temp_x = self.temp_layer1_bn1(temp_x)
            temp_x = self.relu(temp_x)
            temp_x = self.temp_layer1_conv2(temp_x)
            temp_x = self.temp_layer1_bn2(temp_x)
            temp_x = torch.squeeze(self.relu(temp_x))
            # Fusion
            temp_fusion = self.temp_to_spat_layer1(temp_x)
            spat_fusion = self.spat_to_temp_layer1(spat_x)
            temp_x += spat_fusion
            spat_x += temp_fusion

            # Layer2
            spat_x = self.spat_layer2(spat_x)
            temp_x.unsqueeze_(1)
            temp_x = self.temp_layer2_conv1(temp_x)
            temp_x = self.temp_layer2_bn1(temp_x)
            temp_x = self.relu(temp_x)
            temp_x = self.temp_layer2_conv2(temp_x)
            temp_x = self.temp_layer2_bn2(temp_x)
            temp_x = torch.squeeze(self.relu(temp_x))
            # Fusion
            temp_fusion = self.temp_to_spat_layer2(temp_x)
            spat_fusion = self.spat_to_temp_layer2(spat_x)
            temp_x += spat_fusion
            spat_x += temp_fusion
    
            # Layer3
            spat_x = self.spat_layer3(spat_x)
            temp_x.unsqueeze_(1)
            temp_x = self.temp_layer3_conv1(temp_x)
            temp_x = self.temp_layer3_bn1(temp_x)
            temp_x = self.relu(temp_x)
            temp_x = self.temp_layer3_conv2(temp_x)
            temp_x = self.temp_layer3_bn2(temp_x)
            temp_x = torch.squeeze(self.relu(temp_x))
            # Fusion
            temp_fusion = self.temp_to_spat_layer3(temp_x)
            spat_fusion = self.spat_to_temp_layer3(spat_x)
            temp_x += spat_fusion
            spat_x += temp_fusion

            # Layer4
            spat_x = self.spat_layer4(spat_x)
            temp_x.unsqueeze_(1)
            temp_x = self.temp_layer4_conv1(temp_x)
            temp_x = self.temp_layer4_bn1(temp_x)
            temp_x = self.relu(temp_x)
            temp_x = self.temp_layer4_conv2(temp_x)
            temp_x = self.temp_layer4_bn2(temp_x)
            temp_x = torch.squeeze(self.relu(temp_x))
            # Fusion
            temp_fusion = self.temp_to_spat_layer4(temp_x)
            spat_fusion = self.spat_to_temp_layer4(spat_x)
            temp_x += spat_fusion
            spat_x += temp_fusion

            # HEAD
            spat_x += self.head_temp_conv1(temp_x)
            spat_x = self.head_avgpool(spat_x)
            return self.head_fc(torch.squeeze(spat_x))


if __name__ == '__main__':

    TwoStream34 = TwoStream([3, 4, 6, 3])
    x = torch.randn((2, 3, 32, 128, 128))
    print(TwoStream34(x).size())