#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
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
        super(DQN, self).__init__()

        # input arguments
        self.in_channels = in_channels
        self.num_actions = num_actions

        # define variables for each layer
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=8, stride=4, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, self.num_actions)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # normalize
        # x = x/255

        # TODO maybe add maxpooling layers
        # 1st hidden layer: 2d convolution -> relu
        x = self.conv1(x)
        x = self.relu(x)

        # 2nd hidden layer: 2d convolution -> relu
        x = self.conv2(x)
        x = self.relu(x)

        # 3rd hidden layer: 2d convolution -> relu
        x = self.conv3(x)
        x = self.relu(x)

        # flatten convolution layer before fully connected layers
        x = self.flatten(x)

        # 4th hidden layer: fully connected -> relu
        x = self.fc1(x)
        x = self.relu(x)

        # output layer: fully connected
        x = self.fc2(x)

        return x

class DUELING_DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
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
        super(DUELING_DQN, self).__init__()

        # input arguments
        self.in_channels = in_channels
        self.num_actions = num_actions

        # define variables for each layer
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=8, stride=4, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True)

        self.fc1_adv = nn.Linear(7*7*64, 512)
        self.fc1_val = nn.Linear(7*7*64, 512)

        self.fc2_adv = nn.Linear(512, self.num_actions)
        self.fc2_val = nn.Linear(512, 1)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # normalize
        x = x/255

        # 1st hidden layer: 2d convolution -> relu
        x = self.conv1(x)
        x = self.relu(x)

        # 2nd hidden layer: 2d convolution -> relu
        x = self.conv2(x)
        x = self.relu(x)

        # 3rd hidden layer: 2d convolution -> relu
        x = self.conv3(x)
        x = self.relu(x)

        # flatten convolution layer before fully connected layers
        x = self.flatten(x)

        # 4th hidden layer: fully connected -> relu
        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        # output layer: fully connected
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val)

        adv_avg = torch.mean(adv, dim=1, keepdim=True)

        x = val + adv - adv_avg

        return x
