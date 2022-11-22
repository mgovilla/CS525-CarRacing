#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count
import random
from typing import Tuple
import numpy as np
from collections import defaultdict, deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from .agent import Agent
from .dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for replay buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN, self).__init__(env)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.test_dqn:
            # you can load your model here
            print('loading trained model')
            self.trained = True
            map_location = None if torch.cuda.is_available() else torch.device('cpu')
            self.policy_net = torch.load(f="trained_policy_final.pth", map_location=map_location)
            self.policy_net.device = self.device
            self.policy_net.eval()
            return

        if args.train_from_save:
            map_location = None if torch.cuda.is_available() else torch.device('cpu')
            self.policy_net = torch.load(f="trained_policy_final.pth", map_location=map_location)
            self.policy_net.device = self.device
            
            # epsilon
            self.epsilon = 0.025
            self.decay_rate = 0

            # create replay buffer
            self.buffer = deque(maxlen=10000)
            self.batch_size = 64 

            self.init_random_frames = 0
        else: 
            # create the nn model
            self.policy_net = DQN(*env.observation_space.shape,
                                env.action_space.n, device=self.device)
            self.policy_net.to(self.device)

            # epsilon
            self.epsilon = 1.0
            self.epsilon_max_frames = 500000
            
            self.decay_rate = (self.epsilon - 0.025) / (self.epsilon_max_frames)
            
            # create replay buffer
            self.buffer = deque(maxlen=10000)
            self.batch_size = 32 
            self.init_random_frames = 2500

        self.target_net = DQN(*env.observation_space.shape,
                              env.action_space.n, device=self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.to(self.device)
        self.target_net.eval()
        self.trained = False

        # constants
        self.UPDATE_TARGET_FREQ = 2500
        self.gamma = 0.99
        self.max_frames = 1000000
        
        # perform gradient descent every
        self.update_model = 4

    def decay_epsilon(self):
        self.epsilon =  max(self.epsilon - self.decay_rate, 0.025)

    def init_game_setting(self):
        """
        Initialize the replay buffer with some random frames
        """

    def make_action_test(self, observation): 
        with torch.no_grad():
            # get max reward action
            actions = self.policy_net(torch.tensor(np.array([observation.transpose()]), device=self.device, dtype=torch.float))
            a = actions.max(1)[1].view(1, 1).item()
            # print(a, actions)
            return a

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        if test:
            action = self.make_action_test(observation)
            return action

        # epsilon greedy
        p = random.random()
        if p > self.epsilon:
            with torch.no_grad():
                # get max reward action
                actions = self.policy_net(torch.tensor(np.array([observation.transpose()]), device=self.device, dtype=torch.float))
                return actions.max(1)[1].view(1, 1)

        return torch.tensor([[random.randrange(self.env.action_space.n)]], device=self.device, dtype=torch.long)

    def push(self, sars: Tuple):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.

        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        self.buffer.append(sars)

    def replay_buffer(self, batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """

        return random.sample(self.buffer, batch_size)

    def optimize_model(self, optimizer):
        if len(self.buffer) < self.batch_size:
            return

        transitions = random.sample(self.buffer, self.batch_size)
        batch = [*zip(*transitions)]

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch[3])), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch[3]
                                                    if s is not None]).to(self.device)
        state_batch = torch.cat(batch[0]).to(self.device)
        action_batch = torch.cat(batch[1]).to(self.device)
        reward_batch = torch.cat(batch[2]).to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_value_(self.policy_net.parameters(), 1)
        optimizer.step()


    def train(self):
        """
        start with policy_net = target_net
        for iterations:
            play a step in the game using the policy net
            record the step into the buffer

            sample from the buffer 
            compute the reward based on the target_net

            update the policy_net using a loss function (single step) on the reward
            every C iterations, set target_net = policy_net
        """

        rewards_file = open("rewards.out", "w")
        optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00001)
        frames, total_reward, done = 0, 0, True
        while frames < self.max_frames:
            if done:
                state = self.env.reset()
                state = np.array(state[0])
                rewards_file.write(f'{total_reward}\n')
                total_reward = 0

            # decay epsilon after the initial period
            if frames > self.init_random_frames:
                self.decay_epsilon()

            # pick an action
            action = self.make_action(state, False)
            frames += 1

            # play a step in the game based on the policy net
            next_state, reward, done, _, _ = self.env.step(action.item())
            next_state = np.array(next_state)
            total_reward += reward
            # record (s, a, r, s')
            self.push(
                (torch.tensor(np.array([state.transpose()]), device=self.device, dtype=torch.float), 
                action,
                torch.tensor([reward], device=self.device, dtype=torch.float),
                torch.tensor(np.array([next_state.transpose()]), device=self.device, dtype=torch.float) if not done else None))


            state = next_state

            if frames > self.init_random_frames and frames % self.update_model == 0:
                self.optimize_model(optimizer)

            if frames > self.init_random_frames and frames % (self.UPDATE_TARGET_FREQ*self.update_model) == 0:
                # print('updated target net')
                # print('overall average: ', np.average(episode_rewards))
                # all_rewards.extend(episode_rewards)
                # episode_rewards = []
                # print('epsilon: ', self.epsilon)
                # print('replay memory size: ', len(self.buffer))
                rewards_file.flush()
                self.target_net.load_state_dict(self.policy_net.state_dict())
                torch.save(self.policy_net, f"trained_1/trained_policy_{frames}.pth")

        print(frames)
        torch.save(self.policy_net, "trained_policy_final_p2.pth")
