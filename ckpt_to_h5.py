import argparse
from src.madnet import MADNet


parser = argparse.ArgumentParser(description='Script for converting MADNet checkpoint weights to .h5 or vice versa')
parser.add_argument("--search_range", help='maximum dispacement (ie. smallest disparity)',
                    default=2, type=int, required=False)
parser.add_argument("-o", "--output_path",
                    help='Path to save weights in new format (.ckpt or .h5)',
                    required=True)
parser.add_argument("--weights_path",
                    help="Path to pretrained MADNet weights file",
                    required=True)
parser.add_argument("--height", help='model image input height resolution', type=int, default=480)
parser.add_argument("--width", help='model image input height resolution', type=int, default=640)
args = parser.parse_args()


def main(args):
    # Initialise the model
    model = MADNet(
        input_shape=(args.height, args.width, 3),
        weights=args.weights_path,
        search_range=args.search_range
    )

    model.save_weights(args.output_path)


if __name__ == "__main__":
    main(args)