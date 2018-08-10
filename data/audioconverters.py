"""
Classes to convert audio formats.

Required for some processes on linux that cannot load .mp3 formats.
"""

import os
import gc
import sys
import glob

from abc import ABC, abstractmethod

import numpy as np
import librosa


class AudioConverter(ABC):

    """Abstract class implemeting basic audio converter functionality."""

    def __init__(self):
        """Initialize audio converter class."""

    @staticmethod
    def load_librosa(audio_path, sr=22050):
        """
        Load audio file with librosa.

        -----------------------------------------------------------------------

        Args
            track_id: The track_id of the file that will be converted to .wav.

        Return
            audio: Numpy array of the audio loaded by librosa (Ffmpeg)
            sr: Sampling rate used to load the audio
            audio_path: The path to the audio file that was loaded

        -----------------------------------------------------------------------
        """
        audio, sr = librosa.load(audio_path, sr)
        audio = np.expand_dims(audio, 1)
        return audio, sr, audio_path

    @staticmethod
    def convert_path(audio_path, new_ext):
        """
        Convert the audio_path to a path with new_ext.

        -----------------------------------------------------------------------

        Args
            audio_path: The path to the mp3 audio file

        Return
            wav_path: The path to the .wav file saved in the sample place as
            the audio file.

        -----------------------------------------------------------------------
        """
        new_path = audio_path.split('/')
        new_path[-1] = new_path[-1].split('.')[-2]+new_ext
        if not os.path.isdir('/'.join(new_path[:-1])):
            os.mkdir('/'.join(new_path[:-1]))
        new_path = '/'.join(new_path)
        return new_path

    @abstractmethod
    def convert(self):
        """
        Convert the audio to a new format.

        -----------------------------------------------------------------------

        """
        raise NotImplementedError


class WavConverter(AudioConverter):

    """Convert audio file read by librosa to .wav formatself."""

    def __init__(self):
        """Initialize WavConverter."""
        AudioConverter.__init__(self)

    def wav_path(self, audio_path):
        """
        Convert the audio_path to a path for the .wav format.

        -----------------------------------------------------------------------

        Args
            audio_path: The path to the mp3 audio file

        Return
            wav_path: The path to the .wav file saved in the sample place as
            the audio file.

        -----------------------------------------------------------------------
        """
        return self.convert_path(audio_path, ".wav")

    def to_wav(self, audio_path):
        """Convert audio to .wav and save."""
        try:
            audio, sr, audio_path = self.load_librosa(audio_path)
            librosa.output.write_wav(
                self.wav_path(audio_path), audio.astype(float), sr)
        except:
            gc.collect()
            print("Unexpected error:", sys.exc_info()[0])


class WasabiWavConverter(WavConverter):

    """WavConverter for Wasabi database."""

    def __init__(self, audio_dir):
        """
        Initialize WasabiWavConverter.

        Args
            audio_dir: Directory with audio files to convert.
        """
        WavConverter.__init__(self)
        self.audio_dir = audio_dir
        self.track_ids = self._get_track_ids()

    def _get_track_ids(self):
        """
        Collect all of the audio track ids that will be converted.

        -----------------------------------------------------------------------

        Args
            audio_dir: The path to the directory containing the .mp3 files

        Return
            ids: The track ids that were present in the audio_dir.

        -----------------------------------------------------------------------
        """
        ids = []
        for file in glob.glob(os.path.join(self.audio_dir, "*.mp3")):
            if not os.path.isfile(self.wav_path(file)):
                ids += [file.split("/")[-1].split(".")[0]]

        return ids

    def convert(self):
        """Convert audio to .wav and save."""
        for track_id in self.track_ids:
            try:
                audio_path = os.path.join(self.audio_dir, track_id+".mp3")
                self.to_wav(audio_path)
                print("Wavifying track id {}".format(track_id))
            except:
                print("Did not process track id! {}".format(track_id))
