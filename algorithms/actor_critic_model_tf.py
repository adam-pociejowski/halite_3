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


class ActorCriticModelTF(KerasModel):
    def __init__(self, radius, output_number, load_model=False, learning_rate=0.0002):
        super().__init__(output_number=output_number, input_shape=(radius * 2 + 1, radius * 2 + 1, 3))
        self.radius = radius
        self.learning_rate = learning_rate
        self.sess = tf.Session()
        self.build_actor_network()
        self.build_critic_network()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        if load_model:
            self.load()

    def predict(self, X):
        probabilities = self.sess.run(self.outputs_softmax, feed_dict={self.X: X})
        action = np.random.choice(range(len(probabilities.ravel())), p=probabilities.ravel())
        return action

    def train(self):
        if self.episode_step_counter >= 20:
            index_range = min(self.episode_step_counter, self.memory_size)
            sample = np.random.choice(index_range, size=self.batch_size)
            observations_sample = np.zeros((self.batch_size, self.radius * 2 + 1, self.radius * 2 + 1, 3))
            new_observations_sample = np.zeros((self.batch_size, self.radius * 2 + 1, self.radius * 2 + 1, 3))
            rewards_sample = self.reward_memory[sample]
            actions_sample = self.action_memory[sample]
            for i in range(len(sample)):
                observations_sample[i] = self.observation_memory[sample[i]]
                new_observations_sample[i] = self.new_observation_memory[sample[i]]

            action_sample_vectors = []
            for i in range(len(actions_sample)):
                action_vector = np.zeros(self.output_number)
                action_vector[int(actions_sample[i])] = 1
                action_sample_vectors.append(action_vector)

            action_sample_vectors = np.asarray(action_sample_vectors)

            for i in range(self.batch_size):
                new_obs = np.array([new_observations_sample[i]])
                obs = np.array([observations_sample[i]])
                reward = np.array([rewards_sample[i]])[0]
                print(f'new_obs {new_obs.shape} {new_obs}')
                print(f'obs {obs.shape} {obs}')
                print(f'reward {reward.shape} {reward}')
                v = self.sess.run(self.V, {self._X: new_obs})
                print(f'v {v} {v.shape}')
                td_error, _ = self.sess.run([self.td_error, self.critic_train_op], feed_dict={self._X: obs,
                                                                                              self._Y: np.array(v).transpose(),
                                                                                              self._reward: reward})
                print(f'td_error {td_error.shape} {td_error}')

                _, loss = self.sess.run([self.actor_train_op, self.actor_loss], feed_dict={self.X: obs,
                                                                                           self.Y: np.array(action_sample_vectors[i]).reshape((5, 1)),
                                                                                           self.V: td_error})

            # v = self.sess.run(self.V, {self._X: new_observations_sample})
            # td_error, _ = self.sess.run([self.td_error, self.critic_train_op], feed_dict={self._X: observations_sample,
            #                                                                               self._Y: v,
            #                                                                               self._reward: rewards_sample})
            #
            # action_sample_vectors = np.asarray(action_sample_vectors)
            # _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.X: observations_sample,
            #                                                                self.Y: action_sample_vectors,
            #                                                                self.V: td_error})


    # def save(self):
    #     self.actor.save_weights(f"models/actor-{self.model_name}.h5")
    #     self.critic.save_weights(f"models/critic-{self.model_name}.h5")
    #
    # def load(self):
    #     self.actor.load_weights(f"models/actor-{self.model_name}.h5")
    #     self.critic.load_weights(f"models/critic-{self.model_name}.h5")

    def build_actor_network(self):
        shape = (1, self.radius * 2 + 1, self.radius * 2 + 1, 3)
        with tf.name_scope('inputs' + self.model_name):
            self.X = tf.placeholder(tf.float32, shape=shape, name="X" + self.model_name)
            self.Y = tf.placeholder(tf.float32, shape=(self.output_number, None), name="Y" + self.model_name)
            self.V = tf.placeholder(tf.float32, None, name="actions_value" + self.model_name)

        weights = {
            'wc1': tf.get_variable('W0', shape=shape, initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': tf.get_variable('W1', shape=(1, 3, 3, 64), initializer=tf.contrib.layers.xavier_initializer()),
            'wc3': tf.get_variable('W2', shape=(1, 3, 64, 64), initializer=tf.contrib.layers.xavier_initializer()),
            'wd1': tf.get_variable('W3', shape=(64, 128), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('W6', shape=(128, self.output_number), initializer=tf.contrib.layers.xavier_initializer()),
        }
        biases = {
            'bc1': tf.get_variable('B0', shape=(3), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('B4', shape=(self.output_number), initializer=tf.contrib.layers.xavier_initializer()),
        }

        conv1 = self.conv2d(self.X, weights['wc1'], biases['bc1'])
        conv1 = self.maxpool2d(conv1, k=2)

        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = self.maxpool2d(conv2, k=2)

        conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'])
        conv3 = self.maxpool2d(conv3, k=2)

        fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        self.Z3 = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        logits = tf.transpose(self.Z3)
        labels = tf.transpose(self.Y)
        self.outputs_softmax = tf.nn.softmax(self.Z3, name='A3')

        with tf.name_scope('loss' + self.model_name):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            self.actor_loss = tf.reduce_mean(neg_log_prob * self.V)
        with tf.name_scope('train' + self.model_name):
            self.actor_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.actor_loss)

    def build_critic_network(self):
        shape = (1, self.radius * 2 + 1, self.radius * 2 + 1, 3)
        with tf.name_scope('inputs' + self.model_name):
            self._X = tf.placeholder(tf.float32, shape=shape, name="critic_X" + self.model_name)
            self._Y = tf.placeholder(tf.float32, shape=(1, None), name="critic_Y" + self.model_name)
            self._reward = tf.placeholder(tf.float32, None, name='critic_reward' + self.model_name)

        weights = {
            'wc1': tf.get_variable('critic_W0', shape=shape, initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': tf.get_variable('critic_W1', shape=(1, 3, 3, 64), initializer=tf.contrib.layers.xavier_initializer()),
            'wc3': tf.get_variable('critic_W2', shape=(1, 3, 64, 64), initializer=tf.contrib.layers.xavier_initializer()),
            'wd1': tf.get_variable('critic_W3', shape=(64, 128), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('critic_W6', shape=(128, 1), initializer=tf.contrib.layers.xavier_initializer()),
        }
        biases = {
            'bc1': tf.get_variable('critic_B0', shape=(3), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable('critic_B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable('critic_B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable('critic_B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('critic_B4', shape=(1), initializer=tf.contrib.layers.xavier_initializer()),
        }

        conv1 = self.conv2d(self._X, weights['wc1'], biases['bc1'])
        conv1 = self.maxpool2d(conv1, k=2)

        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = self.maxpool2d(conv2, k=2)

        conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'])
        conv3 = self.maxpool2d(conv3, k=2)

        fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        self.V = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

        with tf.variable_scope('critic_loss' + self.model_name):
            self.td_error = self._reward + self.discount_rate * self._Y - self.V
            self.critic_loss = tf.square(self.td_error)
        with tf.variable_scope('critic_train' + self.model_name):
            self.critic_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)

    def conv2d(self, X, W, b, strides=1):
        X = tf.nn.conv2d(X, W, strides=[1, strides, strides, 1], padding='SAME')
        X = tf.nn.bias_add(X, b)
        return tf.nn.relu(X)

    def maxpool2d(self, X, k=2):
        return tf.nn.max_pool(X, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

