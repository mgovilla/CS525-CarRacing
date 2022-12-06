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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .agent import Agent

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

TIME_STEPS = 1000000
BATCH_SIZE = 4800
GAMMA = 0.99
EPSILON = 1e-10
UPDATES_PER_ITER = 5
CLIP = 0.2

class Agent_PPO(Agent):
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

        self.cov_var = torch.full(size=(self.env.action_space.n,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.actor_net = PPO(84, 84, 4, 4, device=self.device)
        self.critic_net = PPO(84, 84, 4, 1, device=self.device)

        self.actor_optim = Adam(self.actor_net.parameters(), lr=LEARNING_RATE)
        self.critic_optim = Adam(self.critic_net.parameters(), lr=LEARNING_RATE)


    def get_action(self, observation):
        mean = self.actor_net(observation)

        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.cpu().detach().numpy(), log_prob.cpu().detach()

    def get_batch(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        ep_rews = []

        _step = 0

        while _step < BATCH_SIZE:
            ep_rews = []

            obs = self.env.reset()
            done = False

            ep_t = 0

            while not done:
                _step += 1
                ep_t += 1

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, reward, done, _ = self.env.step(action)

                ep_rews.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

            batch_lens.append(1 + ep_t)
            batch_rews.append(ep_rews)
        
        batch_obs = torch.tensor(batch_obs, dtype=torch.float, device=self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)

        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in batch_rews[::-1]:
            discounted_reward = 0

            for rew in ep_rews[::-1]:
                discounted_reward = rew + discounted_reward * GAMMA
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=self.device)

        return batch_rtgs

    def get_action(self, observation):
        mean = self.actor_net(observation)

        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

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

        _step = 0
        _eps = 0
        while _step < TIME_STEPS:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.get_batch()

            _step += np.sum(batch_lens)

            V, _ = self.evaluate(batch_obs, batch_acts)
            A = batch_rtgs - V.detach()

            A = (A - A.mean()) / (A.std() + EPSILON)

            _eps += 1

            for _ in range(UPDATES_PER_ITER):
                V, log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(log_probs - batch_log_probs)

                surr1 = ratios * A
                surr2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * A

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            if _eps % 10 == 0:
                torch.save(self.actor_net.state_dict(), './ppo_actor.pth')
                torch.save(self.critic_net.state_dict(), './ppo_critic.pth')
            



