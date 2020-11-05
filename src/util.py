import os
import shutil
import random
import itertools
import numpy as np
import urllib.request
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to)


def generateSilenceSamples(num_samples, _DATASET_DIRECTORY_PATH):
    """Randomly generates a number of 1 second 'silence' samples 
       from the recordings in the _background_noise_ directory.

    Args:
        num_samples ([int]): The total number of silence samples to generate
        _DATASET_DIRECTORY_PATH ([string]): The path to the directory to create and save the silence samples
    """
    if os.path.exists(_DATASET_DIRECTORY_PATH+'/silence'):
        shutil.rmtree(_DATASET_DIRECTORY_PATH+'/silence')
    os.mkdir(_DATASET_DIRECTORY_PATH+'/silence')

    background_noise_sources = [os.path.join(_DATASET_DIRECTORY_PATH+'/_background_noise_', name)
                                for name in os.listdir(_DATASET_DIRECTORY_PATH+'/_background_noise_') if name.endswith('.wav')]

    for index in range(num_samples):
        # Choose random recording
        source_file = random.choice(background_noise_sources)
        (rate, signal) = wav.read(source_file)                # Read audiofile
        # Choose random section
        offset = random.randint(0, len(signal)-rate)
        wav.write(_DATASET_DIRECTORY_PATH+'/silence/{}.wav'.format(index),
                  rate, signal[offset:offset+rate])  # Write new file


def plot_confusion_matrix(cm, class_names):
    """
    From: https://www.tensorflow.org/tensorboard/image_summaries
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)
                       [:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = 0.05
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # color = "black" if cm[i, j] > threshold else "white"
        plt.text(
            j, i, labels[i, j], horizontalalignment="center", fontsize=6, color="white")

    plt.tight_layout()
    plt.grid(False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
