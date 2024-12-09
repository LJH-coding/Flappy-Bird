import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
import time

class FlappyBirdEnv(gym.Env):
    """
    A Gymnasium wrapper for the PLE Flappy Bird game.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(FlappyBirdEnv, self).__init__()

        self.game = FlappyBird()
        self.env = PLE(self.game, fps=30, display_screen=False)

        self.action_space = Discrete(len(self.env.getActionSet()))

        self.observation_space = Box(
            low=0,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset_game()
        state = self._get_state()
        return state, {}

    def step(self, action):
        reward = self.env.act(self.env.getActionSet()[action])
        state = self._get_state()
        done = self.env.game_over()
        info = {}

        return state, reward, done, False, info

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError(f"Render mode {mode} is not supported.")
        self.env.display_screen = True
        time.sleep(0.01)

    def close(self):
        pass

    def _get_state(self):
        state = self.env.getGameState()
        return np.array(list(state.values()), dtype=np.float32)

from gymnasium.envs.registration import register

register(
    id="FlappyBird-v0",
    entry_point="flappybird_env:FlappyBirdEnv"
)
