import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from algorithms.keras_model import KerasModel
import logging
import numpy as np


class DQLModel(KerasModel):
    def __init__(self, radius, output_number, load_model=True, learning_rate=0.001, episode=0):
        super().__init__(model_name='dql', output_number=output_number, input_shape=(radius * 2 + 1, radius * 2 + 1, 3), discount_rate=0.9)
        self.radius = radius
        self.epsilon_decay = 0.01
        self.learning_rate = learning_rate
        self.network = self.build_network()
        self.epsilon = max(0.1, 1.0 - (episode * self.epsilon_decay))
        if load_model:
            self.load()

    def predict(self, X):
        if np.random.random() > self.epsilon:
            prediction = self.network.predict(X)
            actions = np.argmax(prediction, axis=1)
        else:
            actions = [np.random.randint(self.output_number)]

        return actions

    def train(self):
        if self.episode_step_counter >= 50:
            index_range = min(self.episode_step_counter, self.memory_size)
            sample = np.random.choice(index_range, size=self.batch_size)
            observations_sample = np.zeros((self.batch_size, self.radius * 2 + 1, self.radius * 2 + 1, 3))
            new_observations_sample = np.zeros((self.batch_size, self.radius * 2 + 1, self.radius * 2 + 1, 3))
            rewards_sample = self.reward_memory[sample]
            actions_sample = self.action_memory[sample]
            for i in range(len(sample)):
                observations_sample[i] = self.observation_memory[sample[i]]
                new_observations_sample[i] = self.new_observation_memory[sample[i]]

            q_value = self.network.predict(observations_sample, batch_size=self.batch_size)
            q_target = self.network.predict(new_observations_sample, batch_size=self.batch_size)
            batch_indexes = np.arange(self.batch_size, dtype=np.int32)
            actions_indexes = actions_sample.astype(int)

            q_value[batch_indexes, actions_indexes] = rewards_sample + self.discount_rate * np.max(q_target, axis=1)
            self.network.fit(observations_sample, q_value, epochs=1, verbose=0)

    def save(self):
        self.network.save_weights(f"models/dql-{self.model_name}.h5")
        logging.info(f'Weights saved to file: models/dql-{self.model_name}.h5')

    def load(self):
        self.network.load_weights(f"models/dql-{self.model_name}.h5")
        logging.info(f'Weights loaded from file: models/dql-{self.model_name}.h5')

    def build_network(self):
        actor = Sequential()
        actor.add(Conv2D(64, (3, 3), padding='same', input_shape=self.input_shape))
        actor.add(Activation('relu'))
        actor.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        actor.add(Conv2D(64, (3, 3), padding='same'))
        actor.add(Activation('relu'))
        actor.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        actor.add(Conv2D(64, (3, 3), padding='same'))
        actor.add(Activation('relu'))
        actor.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        actor.add(Flatten())
        actor.add(Dense(64))
        actor.add(Activation('relu'))
        actor.add(Dense(self.output_number))
        actor.add(Activation('linear'))
        optimizer = Adam(lr=self.learning_rate)
        actor.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return actor
