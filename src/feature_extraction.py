import os
import util
import shutil
import random
from tqdm import tqdm
import numpy as np
import scipy.io.wavfile as wav
import python_speech_features


# Project Paths
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_DIRECTORY_PATH = _ROOT_DIR+'/data/speech_commands'
_DOWNLOAD_URL = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
_MFCC_SPEECH_COMMANDS_PATH = _ROOT_DIR+'/data/mfcc_speech_commands.npz'
_WORDS = ['unknown', 'silence', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
_MFCC_MAX_SIZE = 16

# Check if directory exist, otherwise download, extract and remove the archive
if not os.path.isdir(_DATASET_DIRECTORY_PATH):
    print('Downloading from ' + _DOWNLOAD_URL)
    util.download_file(_DOWNLOAD_URL, _ROOT_DIR+'/data/speech_commands.tar.gz')
    print("Extracting archive...")
    shutil.unpack_archive(
        _ROOT_DIR+'/data/speech_commands.tar.gz', _DATASET_DIRECTORY_PATH)
    os.remove('data/speech_commands.tar.gz')
    print("Done.")


# Get labels for the dataset
labels = [name for name in os.listdir(_DATASET_DIRECTORY_PATH) if not name.startswith(
    '_') and os.path.isdir(os.path.join(_DATASET_DIRECTORY_PATH, name))]

# Create samples for 'silence' from the longer _background_noises_ recordings
if not 'silence' in labels:
    labels.append('silence')

silence_samples = random.randint(1500, 4000)
util.generateSilenceSamples(silence_samples, _DATASET_DIRECTORY_PATH)

filenames = []
ground_truth = []
for label in labels:
    for file in os.listdir(os.path.join(_DATASET_DIRECTORY_PATH, label)):
        filenames.append(os.path.join(_DATASET_DIRECTORY_PATH, label, file))
        if label in _WORDS:
            ind = _WORDS.index(label)
            ground_truth.append(ind)
        else:
            ground_truth.append(0)  # Reserved for unknown

# Shuffle data
shuffled_data = list(zip(filenames, ground_truth))
random.shuffle(shuffled_data)
filenames, ground_truth = zip(*shuffled_data)

ground_truth = np.asarray(ground_truth).reshape((-1, 1)).squeeze()
one_hot_truth = np.eye(len(_WORDS))[ground_truth]


def generate_mfcc(filename):
    (rate, signal) = wav.read(filename)
    return python_speech_features.base.mfcc(signal, rate,
                                            winlen=0.025,
                                            winstep=0.1)


# Generate MFCC for samples
mfcc_samples = []
for index, filename in enumerate(tqdm(filenames, desc="Generating MFCC for samples")):
    mfcc = generate_mfcc(filename)

    # Add zero-padding when necessary
    padded_mfcc = np.zeros((_MFCC_MAX_SIZE, _MFCC_MAX_SIZE))
    padded_mfcc[:mfcc.shape[0], :mfcc.shape[1]] = mfcc

    # Add axis for color channel
    padded_mfcc = padded_mfcc[:, :, np.newaxis]

    # Normalize
    padded_mfcc = (padded_mfcc - padded_mfcc.min()) / \
        (padded_mfcc.max() - padded_mfcc.min())
    mfcc_samples.append(padded_mfcc)


# Create train, val & test splits
val_set_size = int(len(filenames) * 0.2)
test_set_size = int(len(filenames) * 0.1)

X_val = mfcc_samples[:val_set_size]
X_test = mfcc_samples[val_set_size:(val_set_size + test_set_size)]
X_train = mfcc_samples[(val_set_size + test_set_size):]

# Ground truths
y_val = ground_truth[:val_set_size]
y_test = ground_truth[val_set_size:(val_set_size + test_set_size)]
y_train = ground_truth[(val_set_size + test_set_size):]

# One hot encoded ground truths
Y_val = one_hot_truth[:val_set_size]
Y_test = one_hot_truth[val_set_size:(val_set_size + test_set_size)]
Y_train = one_hot_truth[(val_set_size + test_set_size):]

# Save as compressed numpy array
np.savez(_MFCC_SPEECH_COMMANDS_PATH,
         X_train=X_train,
         Y_train=Y_train,
         y_train=y_train,
         X_val=X_val,
         Y_val=Y_val,
         y_val=y_val,
         X_test=X_test,
         Y_test=Y_test,
         y_test=y_test,
         labels=_WORDS)

print("Saved to " + _MFCC_SPEECH_COMMANDS_PATH)
