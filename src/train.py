import tensorflow as tf
from CNN import CNN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import dirname, abspath
import util

sns.set_style('darkgrid')

# Settings
_ROOT_DIR = dirname(dirname(abspath(__file__)))
_MFCC_SPEECH_COMMANDS_PATH = _ROOT_DIR+'/data/mfcc_speech_commands.npz'
_PATH_TO_RESULTS = _ROOT_DIR+'/results'
_PATH_TO_MODEL = _PATH_TO_RESULTS+'/model/'

_BATCH_SIZE = 100
_EPOCHS = 20
_LEARNING_RATE = 1e-3
_DROPOUT = 0.2

# Load dataset
dataset = np.load(_MFCC_SPEECH_COMMANDS_PATH)

labels = dataset['labels']
X_train = dataset['X_train']
Y_train = dataset['Y_train']
X_val = dataset['X_val']
Y_val = dataset['Y_val']
X_test = dataset['X_test']
Y_test = dataset['Y_test']

model = CNN(num_classes=len(labels), input_shape=(16, 16, 1), n_filters=16,
            learning_rate=_LEARNING_RATE, dropout=_DROPOUT).model

model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    _PATH_TO_MODEL, save_best_only=True, monitor='val_loss', mode='min')

# Reduces learning rate when a metric has stopped improving
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-5, mode='min')

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=_PATH_TO_RESULTS+'/logs', histogram_freq=1)

history = model.fit(X_train, Y_train, epochs=_EPOCHS, batch_size=_BATCH_SIZE, verbose=1,
                    validation_data=(X_val, Y_val),
                    callbacks=[checkpoint, reduce_lr, tensorboard_callback])


tf.saved_model.save(model, _PATH_TO_MODEL)


# Plot results
plt.figure(figsize=(20.0, 10.0))
plt.suptitle('{}'.format(model.name))
plt.subplot(1, 2, 1, label='loss plot')
plt.plot(np.arange(1, len(history.history['loss'])+1), history.history['loss'])
plt.plot(
    np.arange(1, len(history.history['val_loss'])+1), history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.subplot(1, 2, 2, label='dice score plot')
plt.plot(np.arange(
    1, len(history.history['acc'])+1), history.history['acc'])
plt.plot(np.arange(
    1, len(history.history['val_acc'])+1), history.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig(_PATH_TO_RESULTS+'/images/training_process.png')


predictions = model.predict(X_test)
cm = np.asarray(tf.math.confusion_matrix(labels=tf.argmax(
    Y_test, 1), predictions=tf.argmax(predictions, 1), num_classes=len(labels)))
fig = util.plot_confusion_matrix(cm, labels)
fig.savefig(_PATH_TO_RESULTS+'/images/confusion_matrix.png')
