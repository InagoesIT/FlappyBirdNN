import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
from agent import Agent
import time


class Game:
    def __init__(self):
        self.game = FlappyBird()
        self.environment = PLE(self.game)
        self.environment.init()
        self.agent = Agent(allowed_actions=self.environment.getActionSet())
        self.batch_size = 20

    def run(self, episodes):
        self.agent.create_neural_network()
        max_reward = [0, -5]

        for episode in range(episodes):
            self.environment.reset_game()
            episode_reward = 0
            is_game_over = self.environment.game_over()

            while not is_game_over:
                game_state = self.environment.getGameState()
                action = self.agent.choose_action(current_state=game_state)

                # reward = -5 -> dead
                # reward = 1 -> passed through a pipe
                # reward = 0 -> still alive
                reward = self.environment.act(action)
                new_state = self.environment.getGameState()
                is_game_over = self.environment.game_over()
                self.agent.remember(game_state, action, reward, new_state, is_game_over)

                episode_reward += reward

            self.agent.learn(self.batch_size)
            self.batch_size = min((self.batch_size + 20), 100)

            if episode_reward > max_reward[1]:
                max_reward[0] = episode
                max_reward[1] = episode_reward

        print(f"~~~ Max_reward: {max_reward} ~~~~")
