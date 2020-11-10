"""Hyperparameters for the CNN"""

WORDS = ['unknown', 'silence', 'yes', 'no', 'up',
         'down', 'left', 'right', 'on', 'off', 'stop', 'go']

NUM_CLASSES = len(WORDS)
DATASET_SPLIT = (0.7, 0.2, 0.1)
BATCH_SIZE = 100
EPOCHS = 20
LEARNING_RATE = 1e-3
DROPOUT = 0.2

SAMPLE_RATE = 16000
FFT_SIZE = 400
HOP_SIZE = 600
FRAMES = 16
MEL_BINS = 26
MEL_MIN_HZ = 125.0
MEL_MAX_HZ = SAMPLE_RATE / 2
LOG_OFFSET = 1e-3
