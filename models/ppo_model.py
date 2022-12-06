#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch

class PPO(nn.Module):
    def __init__(self, in_size_w, in_size_h, in_channels, num_actions, device="cpu"):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(PPO, self).__init__()
        self.device = device

        # input arguments
        self.in_channels = in_channels
        self.num_actions = num_actions

        # define variables for each layer
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=8, stride=4, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True)
        # calculate input shape for linear layer [(W−K+2P)/S]+1
        conv1_w = (in_size_w - 8) // 4 + 1
        conv1_h = (in_size_h - 8) // 4 + 1

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True)
        conv2_w = (conv1_w - 4) // 2 + 1
        conv2_h = (conv1_h - 4) // 2 + 1

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True)
        conv3_w = (conv2_w - 3) + 1
        conv3_h = (conv2_h - 3) + 1

        self.fc1 = nn.Linear(conv3_w*conv3_h*64, 512)
        self.fc2 = nn.Linear(512, self.num_actions)
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x) # flatten

        x = F.relu(self.fc1(x))
        return self.fc2(x)

