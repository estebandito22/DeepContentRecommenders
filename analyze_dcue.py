"""Script to analyze DCUE model."""

from argparse import ArgumentParser

from analysis.playlist import PlaylistAnalyzer
from dc.nn.dcue import DCUE


def main(model_dir, epoch, layer_number, save_dir):
    """Run analysis."""
    dcue = DCUE()
    pa = PlaylistAnalyzer(dcue, save_dir)
    pa.load(model_dir, epoch)
    pa.analyze(layer_number)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("-m", "--model_dir", help="model directory.")
    ap.add_argument("-e", "--epoch", type=int, help="epoch number.")
    ap.add_argument("-l", "--layer_number", type=int, help="layer number.")
    ap.add_argument("-s", "--save_dir", help="directory to save to.")
    args = vars(ap.parse_args())

    main(args["model_dir"], args["epoch"], args["layer_number"], args["save_dir"])
