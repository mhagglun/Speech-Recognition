import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *


class CNN:

    def __init__(self, num_classes=35, input_shape=(16, 16, 1), n_filters=16, learning_rate=1e-3, dropout=0.2):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.initialization = 'he_normal'
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.model = self.build_model()

    def build_model(self):

        n_filters = self.n_filters

        inputs = Input(self.input_shape)

        conv = Conv2D(n_filters, (5, 5), activation='relu',  kernel_initializer=self.initialization,
                      padding='same')(inputs)

        conv = Conv2D(n_filters, (5, 5), activation='relu',  kernel_initializer=self.initialization,
                      padding='same')(conv)

        pool = MaxPooling2D((2, 2))(conv)

        conv = Conv2D(n_filters*2, (3, 3), activation='relu',  kernel_initializer=self.initialization,
                      padding='same')(pool)

        conv = Conv2D(n_filters*2, (3, 3), activation='relu',  kernel_initializer=self.initialization,
                      padding='same')(conv)

        pool = MaxPooling2D((2, 2))(conv)

        conv = Conv2D(n_filters*4, (2, 2), activation='relu',  kernel_initializer=self.initialization,
                      padding='same')(pool)

        conv = Conv2D(n_filters*4, (3, 3), activation='relu',  kernel_initializer=self.initialization,
                      padding='same')(conv)

        pool = MaxPooling2D((2, 2))(conv)

        pool = Flatten()(pool)
        pool = Dense(256, activation='sigmoid')(pool)
        pool = Dropout(self.dropout)(pool)
        pool = Dense(128, activation='sigmoid')(pool)
        outputs = Dense(self.num_classes, activation='softmax')(pool)

        model = Model(inputs=inputs, outputs=outputs, name='CNN')

        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=['acc'])

        return model
