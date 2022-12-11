from game import Game


'''
RESOURCES USED:
https://github.com/moduIo/Deep-Q-network/blob/master/DDQN.ipynb
https://github.com/GaetanJUVIN/Deep_QLearning_CartPole/blob/master/cartpole.py
https://pygame-learning-environment.readthedocs.io/en/latest/modules/ple.html
https://pygame-learning-environment.readthedocs.io/en/latest/user/games/flappybird.html
https://pygame-learning-environment.readthedocs.io/en/latest/user/home.html
'''

if __name__ == '__main__':
    game = Game()
    game.run(episodes=100)
