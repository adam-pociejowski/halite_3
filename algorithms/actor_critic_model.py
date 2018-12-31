import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from algorithms.keras_model import KerasModel
import logging
import numpy as np


class ActorCriticModel(KerasModel):
    def __init__(self, radius, output_number, load_model=True, learning_rate=0.0001):
        super().__init__(model_name='ml-16_2', output_number=output_number, input_shape=(radius * 2 + 1, radius * 2 + 1, 3))
        self.radius = radius
        self.learning_rate = learning_rate
        self.actor = self.build_actor_network()
        self.critic = self.build_critic_network()
        if load_model:
            self.load()

    def predict(self, X):
        prediction = self.actor.predict(X, batch_size=X.shape[0])
        logging.info(f'prediction: {prediction}')
        return [np.random.choice(self.output_number, 1, p=prediction[i])[0] for i in range(X.shape[0])]

    def train(self):
        if self.episode_step_counter >= 100:
            index_range = min(self.episode_step_counter, self.memory_size)
            sample = np.random.choice(index_range, size=self.batch_size)
            observations_sample = np.zeros((self.batch_size, self.radius * 2 + 1, self.radius * 2 + 1, 3))
            new_observations_sample = np.zeros((self.batch_size, self.radius * 2 + 1, self.radius * 2 + 1, 3))
            rewards_sample = self.reward_memory[sample]
            actions_sample = self.action_memory[sample]
            for i in range(len(sample)):
                observations_sample[i] = self.observation_memory[sample[i]]
                new_observations_sample[i] = self.new_observation_memory[sample[i]]

            value = self.critic.predict(observations_sample)[:, 0]
            next_value = self.critic.predict(new_observations_sample)[:, 0]
            advantages = np.zeros((self.batch_size, self.output_number))
            target = np.zeros((self.batch_size, 1))
            for i in range(self.batch_size):
                advantages[i][int(actions_sample[i])] = rewards_sample[i] + self.discount_rate * next_value[i] - value[i]

            target[:, 0] = rewards_sample + self.discount_rate * next_value
            # logging.info(f'target: {target}')
            # logging.info(f'advantages: {advantages}')
            self.actor.fit(observations_sample, advantages, epochs=1, verbose=0)
            self.critic.fit(observations_sample, target, epochs=1, verbose=0)

    def save(self):
        self.actor.save_weights(f"models/actor-{self.model_name}.h5")
        self.critic.save_weights(f"models/critic-{self.model_name}.h5")

    def load(self):
        self.actor.load_weights(f"models/actor-{self.model_name}.h5")
        self.critic.load_weights(f"models/critic-{self.model_name}.h5")

    def build_actor_network(self):
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
        actor.add(Activation('softmax'))
        optimizer = Adam(lr=self.learning_rate)
        actor.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return actor

    def build_critic_network(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('linear'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

