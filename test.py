from main import Agent, eval
import argparse
import flappybird_env
import gymnasium as gym

def parse_args():
    parser = argparse.ArgumentParser(description='Flappy Bird Q-Learning')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--render', type=lambda x: x.lower() == 'true', default=False, help='Render the environment (True/False)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    env = gym.make('FlappyBird-v0')
    agent = Agent.load(args.checkpoint_dir)
    eval_score, eval_std = eval(agent, env, 100, args.render)
    print(f"Eval Score: {eval_score:.2f} ± {eval_std:.2f}")
