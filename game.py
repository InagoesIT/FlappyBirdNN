import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
from agent import Agent
import time


class Game:
    def __init__(self):
        self.game = FlappyBird()
        self.environment = PLE(self.game, display_screen=True)
        self.environment.init()
        self.agent = Agent(allowed_actions=self.environment.getActionSet())

    def run(self, episodes):
        self.agent.create_neural_network()
        return
        max_reward = [0, -5]

        for episode in range(episodes):
            self.environment.reset_game()
            episode_reward = 0

            while not self.environment.game_over():
                game_state = self.environment.getGameState()
                action = self.agent.choose_action(current_state=game_state)

                # reward = -5 -> dead
                # reward = 1 -> passed through a pipe
                # reward = 0 -> still alive
                reward = self.environment.act(action)
                episode_reward += reward

            if episode_reward > max_reward[1]:
                max_reward[0] = episode
                max_reward[1] = episode_reward

        print(max_reward)
