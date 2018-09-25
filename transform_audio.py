"""Script to transform mp3 to melspectrogram."""

from argparse import ArgumentParser

from data.meltransformer import MelTransformer


def main(raw_dir, save_dir, meta_loc, overwrite, n_fft, hop_length):
    mt = MelTransformer(n_fft, hop_length)
    mt.transform(raw_dir, save_dir, meta_loc, overwrite)


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument("-r", "--raw_dir",
                    help="The directory where raw mp3 files are stored.")
    ap.add_argument("-s", "--save_dir",
                    help="The directory where spectrograms are saved.")
    ap.add_argument("-m", "--meta_loc",
                    help="The full path to the metadata file to save.")
    ap.add_argument("-o", "--overwrite", action='store_true',
                    help="Should previously transformed data be overwritten.")
    ap.add_argument("-n", "--n_fft", default=2048, type=int,
                    help="n_fft to use in melspectrogram transform.")
    ap.add_argument("-hl", "--hop_length", default=512, type=int,
                    help="Hop length to use in melspectrogram transform.")
    args = vars(ap.parse_args())

    rd = args['raw_dir']
    sd = args['save_dir']
    ml = args['meta_loc']
    ov = args["overwrite"]
    nf = args["n_fft"]
    hl = args["hop_length"]

    main(rd, sd, ml, ov, nf, hl)
