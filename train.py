import argparse

from test import test
import time
import gym

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL PRoject4")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--train_from_save', action='store_true', help='whether training from save DQN')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    start_time = time.time()
    env = gym.make("CarRacing-v2", new_step_api=True, continuous=False)
    from models.agent_dqn import Agent_DQN
    agent = Agent_DQN(env, args)
    agent.train()
    print('running time:',time.time()-start_time)
