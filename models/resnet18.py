#!/usr/bin/env python3

import torch
import torch.nn as nn

class ConvBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 has_bn=True,
                 has_act=True):
        super(ConvBnActBlock, self).__init__()
        bias = False if has_bn else True

        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.downsample = True if stride != 1 or inplanes != planes * 1 else False

        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(planes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=False)
        self.relu = nn.ReLU(inplace=True)

        if self.downsample:
            self.downsample_conv = ConvBnActBlock(inplanes,
                                                  planes,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  padding=0,
                                                  groups=1,
                                                  has_bn=True,
                                                  has_act=False)

    def forward(self, x):
        inputs = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.downsample:
            inputs = self.downsample_conv(inputs)

        x = x + inputs
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.downsample = True if stride != 1 or inplanes != planes * 4 else False

        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(planes,
                                    planes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv3 = ConvBnActBlock(planes,
                                    planes * 4,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=False)
        self.relu = nn.ReLU(inplace=True)

        if self.downsample:
            self.downsample_conv = ConvBnActBlock(inplanes,
                                                  planes * 4,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  padding=0,
                                                  groups=1,
                                                  has_bn=True,
                                                  has_act=False)

    def forward(self, x):
        inputs = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample:
            inputs = self.downsample_conv(inputs)

        x = x + inputs
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layer_nums, inplanes=64, num_classes=100):
        super(ResNet, self).__init__()
        self.block = block
        self.layer_nums = layer_nums
        self.num_classes = num_classes
        self.inplanes = inplanes
        self.planes = [inplanes, inplanes * 2, inplanes * 4, inplanes * 8]
        self.expansion = 1 if block is BasicBlock else 4

        self.conv1 = ConvBnActBlock(3,
                                    self.inplanes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)

        self.layer1 = self.make_layer(self.block,
                                      self.planes[0],
                                      self.layer_nums[0],
                                      stride=1)
        self.layer2 = self.make_layer(self.block,
                                      self.planes[1],
                                      self.layer_nums[1],
                                      stride=2)
        self.layer3 = self.make_layer(self.block,
                                      self.planes[2],
                                      self.layer_nums[2],
                                      stride=2)
        self.layer4 = self.make_layer(self.block,
                                      self.planes[3],
                                      self.layer_nums[3],
                                      stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.planes[3] * self.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, planes, layer_nums, stride):
        layers = []
        for i in range(0, layer_nums):
            if i == 0:
                layers.append(block(self.inplanes, planes, stride))
            else:
                layers.append(block(self.inplanes, planes))
            self.inplanes = planes * self.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18(nclasses):
    return ResNet(BasicBlock, [2, 2, 2, 2], 64, nclasses)