import argparse
import tensorflow as tf
from cnn import cnn_tflite_compatible_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="The path to the saved model")
    parser.add_argument(
        "-o", "--output", help="The destination of converted model")
    args = parser.parse_args()

    if args.input is None:
        print("Error: Missing required flag: -i, --input.\nTry 'model_converter --help' for more information. ")
        sys.exit(1)

    if args.output is None:
        print("Error: Missing required flag: -o, --output.\nTry 'model_converter --help' for more information. ")
        sys.exit(1)

    # Load the model and weights
    model = cnn_tflite_compatible_model()
    model.load_weights(args.input)

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    open("{}.tflite".format(args.output), 'wb').write(tflite_model)


if __name__ == "__main__":
    main()
