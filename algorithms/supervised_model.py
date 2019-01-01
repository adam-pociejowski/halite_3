from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import logging


class SupervisedModel:

    def __init__(self, radius, output_number, learning_rate=0.0001):
        self.model_name = 'supervised_cnn_phase1'
        self.learning_rate = learning_rate
        self.output_number = output_number
        self.radius = radius
        self.input_dim = (radius * 2 + 1, radius * 2 + 1, 4)
        self.model = self._create_model()
        self._load_model()
        zeros = np.zeros(self.input_dim)
        self.model.predict(np.array([zeros]))

    def predict(self, X):
        prediction = self.model.predict(X, batch_size=X.shape[0])
        logging.info(f'prediction: {prediction}')
        return [np.random.choice(self.output_number, 1, p=prediction[i])[0] for i in range(X.shape[0])]

    def _load_model(self):
        self.model.load_weights(f'models/{self.model_name}.h5')

    def _create_model(self):
        model = Sequential()
        model.add(Conv2D(128, (3, 3), padding='same', input_shape=self.input_dim))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(self.output_number))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        return model
