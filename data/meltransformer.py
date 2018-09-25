"""Class to transform .mp3 to log-melspectrogram."""

import os
import glob
import warnings
import torch
import pandas as pd
import librosa


class MelTransformer(object):

    """Class to transform .mp3 to log-melspectrogram."""

    def __init__(self, n_fft=2048, hop_length=512):
        """Initialize transformer."""
        self.n_fft = n_fft
        self.hop_length = hop_length

    def transform(self, raw_dir, save_dir, meta_loc, overwrite=False):
        """Transform mp3s to logmelspectrograms."""
        song_ids = []
        song_paths = []
        files = glob.iglob(os.path.join(raw_dir, '*.mp3'))

        print("Globbed files...")
        print("raw dir: {}".format(raw_dir))
        print("save dir: {}".format(save_dir))

        for file in files:
            song_id = file.split("/")[-1].split(".")[0]
            f = os.path.join(save_dir, song_id + "_mel.pt")

            if not os.path.isfile(f) or overwrite:

                try:
                    audio, sr = librosa.load(file, sr=22050)
                    audio = audio[abs(audio) > 0]
                    mel = librosa.feature.melspectrogram(
                        y=audio, sr=sr, n_fft=self.n_fft,
                        hop_length=self.hop_length)
                    mel = librosa.power_to_db(mel)

                    X = torch.from_numpy(mel).float()
                    torch.save(X, f)
                    print("Saving to {}".format(f))
                except:
                    warnings.warn("Could not load file: {}".format(file))

            if os.path.isfile(f):
                song_ids += [song_id]
                song_paths += [f]

        pd.DataFrame(
            {'song_id': song_ids, 'data_mel': song_paths}).to_csv(
                meta_loc, index=False)
