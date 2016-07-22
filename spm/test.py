import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate mask for thumbnail images, distributed')
parser.add_argument("--train", dest='train', help="True for extracting landmark patches, False for ROI (default: %(default)s)", action='store_true')
args = parser.parse_args()

print args.train
