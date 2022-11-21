from test import test
import time
import gym

if __name__ == '__main__':
    start_time = time.time()
    env = gym.make("CarRacing-v2", continuous=False)
    from models.agent_dqn import Agent_DQN
    agent = Agent_DQN(env)
    agent.train()
    print('running time:',time.time()-start_time)

