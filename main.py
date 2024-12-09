import gymnasium as gym
import flappybird_env
import numpy as np
import argparse
import pickle

class Agent:
    def __init__(self, action_space, lr, gamma, epsilon):
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.training = True

    def discretize_state(self, state):
        discrete_state = state
        ### Start Code Here
        ### End Code Here
        return discrete_state
    
    def take_action(self, state):
        if np.random.random() < self.epsilon and self.training:
            return self.action_space.sample()
        else:
            state = self.discretize_state(state)
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.action_space.n)
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        ### Start Code Here
        ### End Code Here

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Flappy Bird Q-Learning')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon Greedy')
    return parser.parse_args()

def eval(agent, env, episodes, render=False):
    total_rewards = []
    agent.training = False
    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0
        while True:
            action = agent.take_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            obs = next_obs
            if done:
                break
        total_rewards.append(ep_reward)
    agent.training = True
    return np.mean(total_rewards), np.std(total_rewards)

def train(agent, env, episodes):
    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        while True:
            action = agent.take_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            if done:
                break
        if episode % 100 == 0:
            eval_score, eval_std = eval(agent, env, 4)
            print(f"Episode {episode}, Eval Score: {eval_score:.2f} ± {eval_std:.2f}")
            agent.save(f'./checkpoints/QLearning_{episode}.pkl')

if __name__ == '__main__':
    args = parse_args()

    env = gym.make("FlappyBird-v0")

    print(env.observation_space)
    """
    observation space detail:
        player y position.
        players velocity.
        next pipe distance to player
        next pipe top y position
        next pipe bottom y position
        next next pipe distance to player
        next next pipe top y position
        next next pipe bottom y position
    """
    print(env.action_space)
    """
    action space detail:
        no_op
        jump
    """

    agent = Agent(action_space=env.action_space, lr=args.lr, gamma=args.gamma, epsilon=args.epsilon)

    train(agent, env, args.episodes)
