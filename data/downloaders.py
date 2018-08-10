"""Downloaders for audio files."""

from urllib.parse import quote
from collections import defaultdict

import os
import json
import time
import requests

import numpy as np
import pandas as pd

from data.utils import rate_limited
from data.Connections import WasabiConnection


class Downloader():

    """Basic Downloader."""

    def __init__(self, save_loc):
        """Initialize downloader."""
        self.save_loc = save_loc

    def download_file(self, url, file):
        """
        Download the specified file to the save_loation directory.

        Args
            url: url where the file is located
            file: the path of the file to save (including the extention)
        """
        file_name = os.path.join(self.save_loc, file)

        if not os.path.isfile(file):
            r = requests.get(url)
            print("Saving {}".format(file_name))
            with open(file, "wb") as f:
                f.write(r.content)

    def save_json(self, obj, file):
        """
        Save object to the specified path in json format.

        Args
            obj: an object to save as json
            file: the path of the file to save (including the extention)
        """
        file_name = os.path.join(self.save_loc, file)

        if not os.path.isfile(file):
            with open(file, "w") as f:
                print("Saving {}.json".format(file_name))
                json.dump(obj, f)


class WasabiDownloader(Downloader):

    """Downloader for the Wasabi music database."""

    def __init__(self, save_loc, found_dir, notfound_dir):
        """
        Initialize Downloader and crate connection to Wasabi API.

        Args
            save_loc: Directory that contains the folders found_dir and
                      notfound_dir where the json and mp3 files will be saved.
            found_dir: Directory name where found song data is saved.
            notfound_dir: Directory name where not found song data is saved.
        """
        Downloader.__init__()
        self.conn = WasabiConnection()
        self.save_loc = save_loc
        self.found_loc = os.path.join(save_loc, found_dir)
        self.notfound_loc = os.path.join(save_loc, notfound_dir)

    @rate_limited(2)
    def _search_artist(self, artist):
        """
        Use the Wasabi API to search for the artistself.

        Finds the best matching result and uses this artist. Rate limited to
        @rate_limit(x)

        Args
            artist: name of the artist.

        Return
            Best matching result on Wasabi database for the given name of
            artist.

        -----------------------------------------------------------------------
        """
        # format artist name according to REST API requirements and make url
        artist = quote(artist, safe='')
        r = self.conn.query("/search/fulltext/"+artist)

        if r.content.find(b'Error') > -1:
            return None
        else:
            r = r.json()

        # return the best matching result found in "namSuggest" field
        if r and 'nameSuggest' in r[0]:
            best_match = r[0]['nameSuggest']['input'][0]
            return best_match

        return None

    @rate_limited(2)
    def _get_artist_data(self, artist):
        """
        Use the Wasabi API to return the artist data provided by Wasabiself.

        For an *exact* artist name that *exists* in the Wasabi database.
        Rate limited to @rate_limit(x)

        Args
            artist: exact artist name that exists in the Wasabi database.

        Return
            artist data json return from the Wasabi API.

        -----------------------------------------------------------------------
        """
        # format artist name according to REST API requirements and make url
        artist = quote(artist, safe='')
        r = self.conn.query("/search/artist/"+artist)
        artist_data = r.json()

        if 'error' in artist_data:
            return None

        return artist_data

    @staticmethod
    def _get_artist_songs(artist_data):
        """
        Collect the songs from a given artist data json.

        Args
            artist_data: artist data json return from the Wasabi API.

        Return
            dataframe of artist, album and songs.

        -----------------------------------------------------------------------
        """
        albums = artist_data['albums']
        songs = defaultdict(list)
        for album in albums:
            for song in album['songs']:

                songs['artist_id'] += [artist_data['_id']]
                songs['artist'] += [artist_data['name']]
                songs['album'] += [album['title']]
                songs['album_id'] += [album['_id']]
                songs['title_lower'] += [str(song['title']).lower()]

                for k, v in song.items():
                    songs[k] += [v]

        return pd.DataFrame(songs)

    @rate_limited(2)
    def _get_song_data(self, song_id):
        """
        Use a Wasabi song id to retun the song data from the Wasabi API.

        Args
            wasabi_song_id: a song id from the Wasabi API.

        Return
            song data json returned from the Wasabi API.

        -----------------------------------------------------------------------
        """
        # call the rest api and convert to json
        success = False
        while success is False:
            r = self.conn.query("/api/v1/song/id/"+song_id)
            if r.content == b'Too many requests, please try again later.':
                success = False
                time.sleep(20)
            else:
                success = True
                song_data = r.json()

        return song_data

    def scrape(self, track_df):
        """
        Download previews and save song data json for tracks from track_df.

        Args
            track_df: Dataframe of the tracks that should be processed.  Must
                      contain the fields "artist", "title_lower", and
                      "song_id".
            save_loc: Directory that contains the folders "clips" and "noclips"
                      where the json and mp3 files will be saved.

        Return
            saves the song data json and mp3 files for previews.

        -----------------------------------------------------------------------
        """
        artists = np.sort(track_df['artist'].dropna().unique())

        # search each artist
        for artist in artists:
            artist = self._search_artist(artist.lower())
            msd_songs = track_df[track_df['artist'] == artist]

            if artist is not None:
                artist_data = self._get_artist_data(artist)

            if artist_data is not None:
                intersect = msd_songs.merge(
                    self._get_artist_songs(artist_data),
                    on=['title_lower']).dropna()

            # add to track dict for each song in both data sets
            if intersect.shape[0] > 0:
                for _, row in intersect.iterrows():

                    mp3_file = "{}.mp3".format(row['song_id'])
                    mp3_exists = os.path.isfile(
                        os.path.join(self.found_loc, mp3_file))

                    json_file = "{}.json".format(row['song_id'])
                    json_exists = os.path.isfile(
                        os.path.join(self.found_loc, json_file))

                    track = self._get_song_data(row['_id'])
                    track['song_id'] = row['song_id']

                    if track['preview']:
                        self.save_loc = self.found_loc
                        if not mp3_exists:
                            self.download_file(track['preview'], mp3_file)
                        if not json_exists:
                            self.save_json(track, json_file)
                    else:
                        self.save_loc = self.notfound_loc
                        print("No preview for {} by {}".
                              format(track['title'], artist))
                        if not json_exists:
                            self.save_json(track, json_file)
