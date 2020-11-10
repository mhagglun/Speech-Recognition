import os
import shutil
import params
import random
import numpy as np
import urllib.request
import tensorflow as tf
import scipy.io.wavfile as wav
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


def generate_log_mel_spectrograms(signal, label):
    """Method for feature extraction. Calculates the log mel-spectrogram representation of the input signal.

    Args:
        signal (tf.Tensor[float]): [description]. The input signal to extract features on.
    Returns:
        [tf.Tensor[float]]: [description] The log mel-spectrogram representation of the input signal
        [tf.Tensor[int]]: [description]   The encoded label of the sample
    """
    # Compute short time fourier transform
    spectrogram = tf.signal.stft(
        signal, frame_length=params.FFT_SIZE, frame_step=params.HOP_SIZE, fft_length=params.FFT_SIZE)

    # Compute magnitudes to avoid complex values
    magnitude_spectrogram = tf.abs(spectrogram)

    # Set the filter bank
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=params.MEL_BINS, num_spectrogram_bins=int(
        params.FFT_SIZE / 2 + 1),    sample_rate=params.SAMPLE_RATE)

    # Transform the linear-scale magnitude-spectrograms to mel-scale
    mel_power_spectrograms = tf.matmul(tf.square(magnitude_spectrogram),
                                       mel_filterbank)

    # Transform magnitudes to log-scale
    log_magnitude_mel_spectrograms = power_to_db(mel_power_spectrograms)
    log_magnitude_mel_spectrograms = tf.expand_dims(
        log_magnitude_mel_spectrograms, axis=-1)

    # MinMax scale the features
    log_magnitude_mel_spectrograms = tf.math.divide(tf.math.subtract(log_magnitude_mel_spectrograms, tf.math.reduce_min(
        log_magnitude_mel_spectrograms)), tf.math.subtract(tf.math.reduce_max(log_magnitude_mel_spectrograms),
                                                           tf.math.reduce_min(log_magnitude_mel_spectrograms)))

    return log_magnitude_mel_spectrograms, label


def power_to_db(S, amin=1e-16, top_db=80.0):
    """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
       Computes the scaling ``10 * log10(S / max(S))`` in a numerically
       stable way.

       Based on:
       http://man.hubwiz.com/docset/LibROSA.docset/Contents/Resources/Documents/generated/librosa.core.power_to_db.html
    """
    def _tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    # Scale magnitude relative to maximum value in S. Zeros in the output
    # correspond to positions where S == ref.
    ref = tf.reduce_max(S)

    log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    return log_spec
