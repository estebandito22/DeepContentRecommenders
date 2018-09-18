"""Script to scrape artist bios from allmusic.com."""

import os
import glob
import json

import pandas as pd

from data.downloaders import AllMusicDownloader


def main(jsons_dir, save_dir, chromedriver_loc):
    """Scrape allmusic.com for artist bios."""
    files = glob.glob(os.path.join(jsons_dir, '*.json'))
    song_id = []
    url_allmusic = []
    for file in files:
        with open(file, 'r') as f:
            metadata = json.load(f)
            song_id += [metadata['_id']]
            url_allmusic += [metadata['urlAllmusic']]

    df = pd.DataFrame({'song_id': song_id, 'url_allmusic': url_allmusic})
    df = df[df['url_allmusic'] != '']

    downloader = AllMusicDownloader(save_dir, chromedriver_loc)
    downloader.scrape(df)


if __name__ == '__main__':
    cl = '/Users/stephencarrow/chromedriver'

    cd = os.getcwd()
    sd = os.path.join(cd, "raw", "bios.csv")

    os.chdir("..")
    os.chdir("DeepSong")
    cd = os.getcwd()
    jd = os.path.join(cd, "data", "MSD", "clips")

    main(jd, sd, cl)
