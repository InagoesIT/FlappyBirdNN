import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
from agent import Agent
import time
import matplotlib.pyplot as plt


class Game:
    def __init__(self):
        self.game = FlappyBird()
        self.environment = PLE(self.game)
        self.environment.init()
        self.agent = Agent(allowed_actions=self.environment.getActionSet())
        self.log_file = "log.txt"

    def run(self, episodes):
        max_reward = [0, -100]
        rewards = []
        total_rewards = 0
        file = open(self.log_file, "a")

        for episode in range(episodes):
            self.environment.reset_game()
            episode_reward = 0
            is_game_over = self.environment.game_over()
            start = time.time()

            while not is_game_over:
                game_state = self.environment.getGameState()
                action = self.agent.choose_action(current_state=game_state)

                # reward = -5 -> dead
                # reward = 1 -> passed through a pipe
                # reward = 0 -> still alive
                reward = self.environment.act(action)
                if reward == -5:
                    reward = -100
                new_state = self.environment.getGameState()
                is_game_over = self.environment.game_over()
                self.agent.remember(game_state, action, reward, new_state, is_game_over)

                episode_reward += reward

            self.agent.learn()
            rewards.append(episode_reward)

            if episode_reward > max_reward[1]:
                max_reward[0] = episode
                max_reward[1] = episode_reward

            total_rewards += episode_reward

            log_text = f"-----> episode ended {episode} and lasted {time.time() - start} and got the reward {episode_reward}" \
                       f" with epsilon {self.agent.epsilon}\n"
            file.write(log_text)
            print(log_text)

        file.close()
        self.agent.save_model()
        print(f"~~~ Max_reward: {max_reward} ~~~~")

        file.write(f"Total rewards for {episodes} {total_rewards}")
        print(f"Total rewards for {episodes} {total_rewards}")

        plt.plot(range(episodes), rewards)
        plt.ylabel("rewards")
        plt.xlabel("episodes")
        plt.legend()
        plt.savefig("graph.png")

