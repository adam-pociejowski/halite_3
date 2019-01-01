import tensorflow as tf
import os
import numpy as np
import abc
import logging
import datetime
import time
ts = time.time()


class KerasModel:
    def __init__(self, model_name, output_number=5, input_shape=(3, 3, 3), memory_size=1000, batch_size=20, min_memory_size=2000, discount_rate=0.99):
        self.output_number = output_number
        self.input_shape = input_shape
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.min_memory_size = min_memory_size
        __metaclass__ = abc.ABCMeta
        self._init_episode()
        self.model_name = model_name

    def post_step_actions(self, observations, actions, rewards, new_observations):
        self._store_memory(observations, actions, rewards, new_observations)
        if self.episode_step_counter > self.min_memory_size:
            self._train()
        super().post_step_actions(observations, actions, rewards, new_observations)

    def post_episode_actions(self, rewards, episode):
        self._delete_objects()
        self._init_episode()

    def _get_memory_samples(self):
        index_range = min(self.episode_step_counter, self.memory_size)
        sample = np.random.choice(index_range, size=self.batch_size)
        return self.observation_memory[sample, :], self.new_observation_memory[sample, :], self.reward_memory[sample], self.action_memory[sample]

    def _init_episode(self):
        self.action_memory = np.zeros(self.memory_size)
        self.reward_memory = np.zeros(self.memory_size)
        self.observation_memory = {}
        self.new_observation_memory = {}
        self.episode_step_counter = 0

    def store_memory(self, observations, actions, rewards, new_observations):
        index = self.episode_step_counter % self.memory_size
        self.action_memory[index] = actions
        self.reward_memory[index] = rewards
        self.observation_memory[index] = observations
        self.new_observation_memory[index] = new_observations
        self.episode_step_counter += 1

    def _delete_objects(self):
        del self.action_memory
        del self.reward_memory
        del self.observation_memory
        del self.new_observation_memory
