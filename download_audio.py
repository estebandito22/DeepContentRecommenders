"""Script to download audio files."""

import os
import glob

import numpy as np
import pandas as pd

from data.downloaders import WasabiDownloader


def get_visited_song_ids(msd_dir):
    """
    Build a list of the previously processed MSD song idsself.

    Used so that data collection can resume if it is interrupted.

    Args
        msd_dir: Directory that contains the clips and the noclips folders
        where the mp3 and song data json are saved.

    Return
        list of the MSD song ids that have already been processed.

    ---------------------------------------------------------------------------
    """
    ids = []
    for f in glob.glob(os.path.join(msd_dir, "clips", "*.json")):
        ids += [f.split("/")[-1].split(".")[0]]

    for f in glob.glob(os.path.join(msd_dir, "noclips", "*.json")):
        ids += [f.split("/")[-1].split(".")[0]]

    return ids


if __name__ == "__main__":

    CUR_DIR = os.getcwd()

    # load taste profile dataset
    file = os.path.join(CUR_DIR, "data", "MSD", "train_triplets.txt")
    colnames = ['user_id', 'song_id', 'playcount']
    taste_df = pd.read_table(file, header=None, names=colnames)

    # load msd track info
    file = os.path.join(CUR_DIR, "data", "MSD", "unique_tracks.txt")
    colnames = ['track_id', 'song_id', 'artist_name', 'song_title']
    track_df = pd.read_table(file, header=None, names=colnames,
                             sep='<SEP>', engine='python')

    # combine track info with taste profile
    # required to search spotify
    taste_df = taste_df.merge(track_df, on=['song_id'], how='left')

    # get unique tracks from the taste profile dataset
    tracks_taste_df = taste_df[
        ['song_id', 'track_id', 'artist_name', 'song_title']].drop_duplicates()
    tracks_taste_df.columns = ['song_id', 'track_id', 'artist', 'title']
    tracks_taste_df['title_lower'] = tracks_taste_df[
        'title'].apply(lambda x: str(x).lower())

    # dont process songs that have already been visited
    visited_ids = get_visited_song_ids(os.path.join(CUR_DIR, "data", "MSD"))
    visited_mask = np.where(~tracks_taste_df['song_id'].isin(visited_ids))[0]
    tracks_taste_df = tracks_taste_df.iloc[visited_mask]

    # get spotify preview urls and save
    save_loc = os.path.join(CUR_DIR, "audio", "MSD")
    wasabi_downloader = WasabiDownloader(save_loc, 'clips', 'noclips')
    wasabi_downloader.scrape(tracks_taste_df)
