import sys
import argparse
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="The path to the saved model")
    parser.add_argument(
        "-o", "--output", help="The destination of converted model")
    args = parser.parse_args()

    if args.input is None:
        print("Error: Missing required flag: -i, --input.\nTry 'model-converter --help' for more information. ")
        sys.exit(1)

    if args.output is None:
        print("Error: Missing required flag: -o, --output.\nTry 'model-converter --help' for more information. ")
        sys.exit(1)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=args.input)
    lite_model = converter.convert()
    open(args.output, 'wb').write(lite_model)
    

if __name__ == "__main__":
    main()