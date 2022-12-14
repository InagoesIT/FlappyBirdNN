from collections import deque

import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Agent:
    def __init__(self, allowed_actions):
        # actions encoding : [119, None] => ["UP", "NOTHING"]
        self.go_up = allowed_actions[0]
        self.do_nothing = allowed_actions[1]
        self.actions = [self.do_nothing, self.go_up]

        # Q-learning hyperparameters
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1  # Exploration rate
        self.epsilon_min = 0.1  # Minimal exploration rate
        self.epsilon_decay = 0.99  # Decay rate for epsilon

        self.weight_backup_file = "weights.h5"
        self.model = Agent.create_neural_network(self.weight_backup_file)
        self.memory = deque(maxlen=30000)

        self.epochs = 2
        self.batch_size = 100

    '''
    current_state:
        player y position.
        players velocity.
        next pipe distance to player
        next pipe top y position
        next pipe bottom y position
        next next pipe distance to player
        next next pipe top y position
        next next pipe bottom y position

    '''

    @staticmethod
    def get_encoded_state(state):
        return [value for value in state.values()]

    def choose_action(self, current_state: dict):
        if np.random.rand() <= self.epsilon:
            if np.random.rand() < 0.5:
                return self.go_up
            return self.do_nothing
        encoded_state = Agent.get_encoded_state(current_state)
        q_values = self.model.predict(np.array([encoded_state]), batch_size=1, verbose=0)
        return self.actions[np.argmax(q_values)]

    def save_model(self):
        self.model.save_weights(self.weight_backup_file)

    def remember(self, state, action, reward, next_state, is_game_over):
        action_index = 1
        if action == self.do_nothing:
            action_index = 0
        encoded_state = Agent.get_encoded_state(state)
        encoded_next_state = Agent.get_encoded_state(next_state)

        self.memory.append((encoded_state, action_index, reward, encoded_next_state, is_game_over))

    @staticmethod
    def create_neural_network(weights_path) -> keras.Sequential:
        model = keras.Sequential()
        model.add(layers.Input(8, name="layer1"))
        model.add(layers.Dense(32, activation="relu", name="layer2"))
        model.add(layers.Dense(18, activation="relu", name="layer3"))
        model.add(layers.Dense(2, activation=None, name="layer4"))

        model.compile(optimizer='rmsprop',
                      loss='mse',
                      metrics=['accuracy'])

        model.load_weights(weights_path)

        return model

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        sample_batch = random.sample(self.memory, self.batch_size)

        states = []
        targets = []

        for state, action_index, reward, next_state, is_game_over in sample_batch:
            opposite_action_index = 1 - action_index
            target = [0, 0]
            predicted = self.model.predict(np.array([state]), verbose=0)
            target[opposite_action_index] = float(predicted[0][opposite_action_index])
            target[action_index] = reward

            if not is_game_over:
                target[action_index] = reward + self.gamma * np.max(self.model.predict(np.array([next_state]), verbose=0))

            states.append(list(state))
            targets.append(list(target))

        self.model.fit(states, targets, self.epochs)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
