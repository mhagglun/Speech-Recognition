# Speech Recognition with TensorFlow

In this project a Convolutional Neural Network is implemented using TensorFlow in order to perform speech recognition. Additionally, inference will be run on the trained model using TensorFlow Lite to obtain a smaller model that is suitable for being deployed on a Raspberry Pi.

## Overview

- [Speech Recognition with TensorFlow](#speech-recognition-with-tensorflow)
  - [Overview](#overview)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Usage](#usage)
  - [Dataset](#dataset)
  - [Pipeline](#pipeline)
    - [Preprocessing](#preprocessing)
    - [Data Augmentation](#data-augmentation)
    - [Feature Extraction](#feature-extraction)
  - [Training](#training)
  - [Deploy Model on Raspberry Pi](#deploy-model-on-raspberry-pi)

## Getting Started

### Installation

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Usage

Start the notebook `train.ipynb` which will download the dataset, set up the pipeline for preprocessing and train the model.

```
notebook jupyter
```

The trained model which is saved to `results/model.h5` can then be converted into the TFLite version

```
python src/model_converter.py -i results/model.h5 -o results/model
```

## Dataset

The dataset used is the Speech Commands from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/speech_commands), containing a total of more than 100k recordings of 35 spoken words.

The distribution of samples among the labels are shown below
![](docs/images/sample_distribution.png)

The trained network will be trained to identify 12 categories, 10 of which are words and the remaining two is whether the sound is either `unknown` or `silence`. Specifically, the labels are

```
'unknown', 'silence', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'
```

Because the remaining samples are pooled into the new category `unknown` the sample distribution becomes very imbalanced.

![](docs/images/pooled_sample_distribution.png)

In order to deal with the problem of imbalanced classes we'll calculate and use _class_weights_ to use during training.
The weights will penalize mistakes in made for samples of `class[i]` with `class_weight[i]` instead of 1. So higher class-weight means you want to put more emphasis on a class.

## Pipeline

A pipeline for preprocessing and data augmentation is set up in order to train the network.

### Preprocessing

The preprocessing consists of decoding audio samples (.wav-file) such that the amplitudes are `[-1.0, 1.0]`.
Since each audio sample is only approximately 1 second long, the sizes will vary slightly. Each sample is therefore zero-padded to insure that the inputs always have equal shape.

### Data Augmentation

Training samples will randomly be augmented by adding noise to the source signal. The noise added consists of random segments from one of the longer recordings from `/_background_noise_`.

### Feature Extraction

The features that the network will train on are the log melspectograms calculated from the signals.

A log mel spectrogram is generated by

- Transforming the signal from time domain to frequency domain using Short-Time Fourier Transform (STFT).
  Tells us the magnitude of each frequency present in the entire signal.
- The spectrogram is generated by applying the STFT over windows of the entire signal.
  It gives the distribution of frequencies for each instant of time.
- Transform to Mel-Scale. It's a logarithmic transform which is done to scale frequencies in a way that's relevant to humans.
  For instance, there's a larger perceivable difference between 500Hz to 1000Hz than 7500Hz to 8000Hz in comparison.
- After converting the frequency spectrogram to Mel scale we get the _Mel-Spectrogram_!
  The values become more meaningful by taking the logarithm of them to get them to decibels. This is the log scaled Mel-Spectrogram.

The actual feature extraction is implemented in such a way that it's not part of the data pipeline per se. Instead a custom layer is added as the first layer of the network where the features are computed. This means that the feature extraction is done by the GPU instead of the CPU during the preprocessing stage, which results in a considerable reduction in total training time.

Below are some visual representations of the features (Scaled log melspectograms) extracted for some of the spoken words, which the CNN will learn from.

![](docs/images/extracted_features.png)

## Training

The network was trained for 20 epochs using `dropout=0.2` and `batch_size=100`, with an initial learning rate of 1e-3.

![](results/images/training_process.png)
![](results/images/confusion_matrix.png)

## Deploy Model on Raspberry Pi

Once the final model has been trained and converted using Tensorflow Lite, the TFLite model can be evaluated on the Raspberry Pi with

```
python src/classipier.py -f results/model.tflite -d <PATH TO TEST DIRECTORY>
```
