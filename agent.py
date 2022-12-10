import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Agent:
    def __init__(self, allowed_actions):
        # [119, None] => ["UP", "NOTHING"]
        self.go_up = allowed_actions[0]
        self.do_nothing = allowed_actions[1]

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

    def choose_action(self, current_state: dict):
        if random.randint(0, 1) < 0.5:
            return self.go_up
        return self.do_nothing

    def create_neural_network(self):
        model = keras.Sequential()
        model.add(layers.Input(8, name="layer1"))
        model.add(layers.Dense(80, activation="relu", name="layer2"))
        model.add(layers.Dense(150, activation="relu", name="layer3"))
        model.add(layers.Dense(2, activation=None, name="layer4"))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        # # Call layers on a test input
        # x = tf.ones((3, 3))
        # y = model(x)
