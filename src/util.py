from numpy.core.fromnumeric import size
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import itertools
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
