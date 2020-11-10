import os
import glob
import params
import librosa
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite

from tqdm import tqdm

_LABELS = np.array(params.WORDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filepath", help="The path to the saved model")
    parser.add_argument(
        "-d", "--dirname", help="The path to the test directory")
    args = parser.parse_args()

    if args.filepath is None:
        print("Error: Missing required flag: -f, --filepath.\nTry 'model-converter --help' for more information. ")
        sys.exit(1)

    if args.dirname is None:
        print("Error: Missing required flag: -d, --dirname.\nTry 'model-converter --help' for more information. ")
        sys.exit(1)

    interpreter = tflite.Interpreter(model_path=args.filepath)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    data_dir = pathlib.Path(args.dirname)
    filenames = glob.glob(str(data_dir) + '/*/*')
    true_labels = []
    predicted_labels = []

    for filename in tqdm(filenames, desc="Predicting"):
        signal, label = get_sample(filename)
        interpreter.set_tensor(
            input_details[0]['index'], signal)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        label_id = np.argmax(output_data)

        predicted_labels.append(label_id)
        true_labels.append(np.where(label == _LABELS)[0][0])

    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    accuracy = (true_labels == predicted_labels).mean()
    print("Accuracy on test set: {}".format(accuracy))


def get_sample(filepath):
    signal, _ = librosa.load(filepath, sr=params.SAMPLE_RATE)
    signal = np.reshape(signal, (1, params.SAMPLE_RATE))

    label = filepath.split('/')[-2]
    return signal, label


if __name__ == "__main__":
    main()
