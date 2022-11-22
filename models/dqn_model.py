#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch

class DQN(nn.Module):
    """Initialize a deep Q-learning network
    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_size_w, in_size_h, in_channels=4, num_actions=4, device="cpu"):
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
        self.device = device

        # Network defined by the Deepmind paper

        # Convolutions on the frames on the screen
        # [(W−K+2P)/S]+1
        self.layer1 = nn.Conv2d(in_channels, 32, 8, 4)
        conv1_w = (in_size_w - 8) // 4 + 1
        conv1_h = (in_size_h - 8) // 4 + 1
        
        self.layer2 = nn.Conv2d(32, 64, 4, 2)
        conv2_w = (conv1_w - 4) // 2 + 1
        conv2_h = (conv1_h - 4) // 2 + 1

        self.layer3 = nn.Conv2d(64, 64, 3, 1)
        conv3_w = (conv2_w - 3) + 1
        conv3_h = (conv2_h - 3) + 1

        self.layer4 = nn.Flatten()

        self.layer5 = nn.Linear(64 * conv3_w * conv3_h, 512)
        self.action = nn.Linear(512, num_actions)


        # KERNEL_SIZE=3
        # self.conv1 = nn.Conv2d(in_channels, 16, KERNEL_SIZE)
        # conv1_w = in_size_w - KERNEL_SIZE + 1
        # conv1_h = in_size_h - KERNEL_SIZE + 1

        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(16*conv1_w*conv1_h, 512)
        # self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = x.to(self.device)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x) # flatten

        x = F.relu(self.layer5(x))
        return self.action(x)
        