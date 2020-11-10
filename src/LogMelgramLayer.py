import tensorflow as tf
import features_tflite as features_tflite_lib


class LogMelgramLayer(tf.keras.layers.Layer):

    def __init__(
        self, sample_rate, num_fft, hop_length, num_mels, f_min=125.0, f_max=3800.0, ** kwargs
    ):
        super(LogMelgramLayer, self).__init__(**kwargs)
        self.num_fft = num_fft
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.num_spectrogram_bins = num_fft // 2 + 1
        mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mels,
            num_spectrogram_bins=self.num_spectrogram_bins,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max,
        )

        self.mel_filterbank = mel_filterbank

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super(LogMelgramLayer, self).build(input_shape)

    def call(self, input):
        """
        Args:
            input (tensor): Batch of mono waveform, shape: (None, N)
        Returns:
            log_melgrams (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins, channel=1)
        """

        def power_to_db(S, amin=1e-16, top_db=80.0):
            """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
            Computes the scaling ``10 * log10(S / max(S))`` in a numerically
            stable way.

            Based on:
            http://man.hubwiz.com/docset/LibROSA.docset/Contents/Resources/Documents/generated/librosa.core.power_to_db.html
            """
            def _tf_log10(x):
                numerator = tf.math.log(x)
                denominator = tf.math.log(
                    tf.constant(10, dtype=numerator.dtype))
                return numerator / denominator

            # Scale magnitude relative to maximum value in S. Zeros in the output
            # correspond to positions where S == ref.
            ref = tf.reduce_max(S)

            log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
            log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

            return log_spec

        # Compute short time fourier transform
        spectrogram = tf.signal.stft(
            input, frame_length=self.num_fft, frame_step=self.hop_length, fft_length=self.num_fft)

        # Compute magnitudes to avoid complex values
        magnitude_spectrogram = tf.abs(spectrogram)

        # Transform the linear-scale magnitude-spectrograms to mel-scale
        mel_power_spectrograms = tf.matmul(tf.square(magnitude_spectrogram),
                                           self.mel_filterbank)

        # Transform magnitudes to log-scale
        log_magnitude_mel_spectrograms = power_to_db(
            mel_power_spectrograms)
        log_magnitude_mel_spectrograms = tf.expand_dims(
            log_magnitude_mel_spectrograms, axis=-1)

        # MinMax scale the features
        log_magnitude_mel_spectrograms = tf.math.divide(tf.math.subtract(log_magnitude_mel_spectrograms, tf.math.reduce_min(
            log_magnitude_mel_spectrograms)), tf.math.subtract(tf.math.reduce_max(log_magnitude_mel_spectrograms),
                                                               tf.math.reduce_min(log_magnitude_mel_spectrograms)))
        return log_magnitude_mel_spectrograms

    def get_config(self):
        config = {
            'num_fft': self.num_fft,
            'hop_length': self.hop_length,
            'num_mels': self.num_mels,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        base_config = super(LogMelgramLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
