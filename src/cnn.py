import os
import sys
import params
import features_tflite as features_tflite_lib
from LogMelgramLayer import LogMelgramLayer
from tensorflow.keras import Model, Sequential, layers


def cnn():
    """Defines the core model"""
    model = Sequential(layers=[
        LogMelgramLayer(input_shape=((params.SAMPLE_RATE,)), sample_rate=params.SAMPLE_RATE,
                        num_fft=params.FFT_SIZE, hop_length=params.HOP_SIZE, num_mels=params.MEL_BINS),
        layers.Conv2D(16, 3, activation='relu',
                      kernel_initializer='he_normal'),
        layers.Conv2D(32, 2, activation='relu',
                      kernel_initializer='he_normal'),
        layers.Conv2D(64, 2, activation='relu',
                      kernel_initializer='he_normal'),
        layers.Conv2D(128, 2, activation='relu',
                      kernel_initializer='he_normal'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(params.DROPOUT),
        layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
        layers.Dense(len(params.WORDS), activation='softmax'),
    ], name="CNN")

    return model


def cnn_tflite_compatible_model():

    waveform = layers.Input(batch_shape=(1, params.SAMPLE_RATE))
    logMelgram = features_tflite_lib.waveform_to_log_mel_spectrogram(
        waveform, params)

    net = layers.Reshape(
        (params.FRAMES, params.MEL_BINS, 1))(logMelgram)

    net = layers.Conv2D(16, 3, activation='relu',
                        kernel_initializer='he_normal')(net)
    net = layers.Conv2D(32, 2, activation='relu',
                        kernel_initializer='he_normal')(net)
    net = layers.MaxPooling2D()(net)
    net = layers.Conv2D(64, 2, activation='relu',
                        kernel_initializer='he_normal')(net)
    net = layers.MaxPooling2D()(net)
    net = layers.Flatten()(net)
    net = layers.Dense(128, activation='relu',
                       kernel_initializer='he_normal')(net)
    net = layers.Dropout(params.DROPOUT)(net)
    net = layers.Dense(len(params.WORDS), activation='softmax')(net)

    model = Model(name="TFLITE_COMPATIBLE_CNN",
                  inputs=waveform, outputs=net)
    return model
