from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch.autograd import Variable
import datetime

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ShuffleUnit(nn.Module):
    def __init__(self, in_channel, out_channel, stride, split_ratio, groups):
        super(ShuffleUnit, self).__init__()

        self.stride = stride
        self.groups = groups

        if 1 != self.stride:
            self.relu  = nn.ReLU6(inplace=True)
            self.rconv1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False)
            self.rbn1   = nn.BatchNorm2d(in_channel)
            self.rconv2 = nn.Conv2d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False)
            self.rbn2   = nn.BatchNorm2d(in_channel)
            self.rconv3 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False)
            self.rbn3   = nn.BatchNorm2d(in_channel)

            self.lconv1 = nn.Conv2d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False)
            self.lbn1   = nn.BatchNorm2d(in_channel)
            self.lconv2 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False)
            self.lbn2   = nn.BatchNorm2d(in_channel)
        else:
            # Channel Split
            self.left_in_channel = int(split_ratio * in_channel)
            #print('in_channel', in_channel)
            #print('left_in_channel: ', self.left_in_channel)
            right_in_channel = in_channel - self.left_in_channel
            right_out_channel = out_channel - self.left_in_channel

            self.relu  = nn.ReLU6(inplace=True)
            self.rconv1 = nn.Conv2d(right_in_channel, right_out_channel, 1, 1, 0, bias=False)
            self.rbn1   = nn.BatchNorm2d(right_out_channel)
            self.rconv2 = nn.Conv2d(right_out_channel, right_out_channel, 3, stride, 1, groups=right_out_channel, bias=False)
            self.rbn2   = nn.BatchNorm2d(right_out_channel)
            self.rconv3 = nn.Conv2d(right_out_channel, right_out_channel, 1, 1, 0, bias=False)
            self.rbn3   = nn.BatchNorm2d(right_out_channel)

    def forward(self, x):
        #print(x.size())
        if 1 != self.stride:
            r_x = self.relu(self.rbn1(self.rconv1(x)))
            r_x = self.rbn2(self.rconv2(r_x))
            r_x = self.relu(self.rbn3(self.rconv3(r_x)))

            l_x = self.lbn1(self.lconv1(x))
            l_x = self.relu(self.lbn2(self.lconv2(l_x)))
            #print(l_x.size(), r_x.size())
        else:
            l_x = x[:, 0:self.left_in_channel, :, :]
            r_x = x[:, self.left_in_channel:, :, :]

            r_x = self.relu(self.rbn1(self.rconv1(r_x)))
            r_x = self.rbn2(self.rconv2(r_x))
            r_x = self.relu(self.rbn3(self.rconv3(r_x)))
            #print(l_x.size(), r_x.size())

        x_out = torch.cat((l_x, r_x), 1)
        x_shuffle = channel_shuffle(x_out, self.groups)
        #print(x_shuffle.size())
        return x_shuffle

class ShuffleNetV2(nn.Module):
    def __init__(self, scale=1.0, split_ratio=0.5, groups=2, num_classes=1000):
        super(ShuffleNetV2, self).__init__()

        self.num_of_channels = {0.5: [24, 48, 96, 192, 1024], 1: [24, 116, 232, 464, 1024],
                                1.5: [24, 176, 352, 704, 1024], 2: [24, 244, 488, 976, 2048]}
        self.c = [_make_divisible(chan, groups) for chan in self.num_of_channels[scale]]
        self.blocks = [4, 8, 4] 
        self.conv1 = nn.Conv2d(3, self.c[0], 3, 2, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(self.c[0])
        self.maxpool = nn.MaxPool2d(3, 2)

        self.stage2 = self._make_layer(self.blocks[0], self.c[0], self.c[1], split_ratio, groups)
        self.stage3 = self._make_layer(self.blocks[1], self.c[1], self.c[2], split_ratio, groups)
        self.stage4 = self._make_layer(self.blocks[2], self.c[2], self.c[3], split_ratio, groups)

        self.conv5 = nn.Conv2d(self.c[3], self.c[4], 1, 1, 0, bias=False)
        self.bn5   = nn.BatchNorm2d(self.c[4])
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.c[4], num_classes)

        self.relu = nn.ReLU6(inplace=True)

        self._initialize_weights()

    def _make_layer(self, blocks, in_channels, out_channels, split_ratio, groups):
        strides = [2] + [1] * (blocks - 1)
        layers = []
        for _stride in strides:
            layers.append(ShuffleUnit(in_channels, out_channels, _stride, split_ratio, groups))
            if 2 == _stride:
                in_channels = 2 * in_channels
            else:
                in_channels = out_channels
        
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_() 
        
    def forward(self, x):
        x_out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x_out = self.stage2(x_out)
        x_out = self.stage3(x_out)
        x_out = self.stage4(x_out)
        x_out = self.gap(self.relu(self.bn5(self.conv5(x_out))))
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.classifier(x_out)
        return x_out

if __name__ == "__main__":
    """Testing
    """
    net = ShuffleNetV2()
    x = torch.randn(1, 3, 224, 224)
    #y = net(x)
    #print(net)

    for i in range(15):
        time1 = datetime.datetime.now()
        y = net(x)
        print('Time Cost: ', (datetime.datetime.now() - time1).microseconds)
