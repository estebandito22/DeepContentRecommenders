"""Classes to connect to music databse APIs."""

import base64
import requests

from selenium import webdriver

from data.utils import rate_limited


class Connection():

    """Base connetion class."""

    def __init__(self, base_url):
        """
        Inialize base connection.

        Args
            base_url: The base url of the API to connect to.
        """
        self.base_url = base_url

    @rate_limited(2)
    def query(self, query):
        """
        Execute an API query.

        query: String containing RESTful API query.
        """
        return requests.get(self.base_url + query)


class SpotifyConnection(Connection):

    """
    Class object instantiates a connection with spotify.

    When the connection is alive, queries are made with the query_get method.

    Courtesy of https://github.com/ritchie46/music-classification
    """

    def __init__(self, client_id, secret):
        """
        Initialize Spotify API connection.

        Args
            client_id: client_id provided by Spotify API.
            secret: secret key provided by Spotify API.
        """
        Connection.__init__(self, "https://api.spotify.com")

        # First header and parameters needed to require an access token.
        param = {"grant_type": "client_credentials"}
        header = {"Authorization": "Basic {}".format(
            base64.b64encode("{}:{}".format(client_id, secret)
                             .encode("ascii")).decode("ascii")),
                  'Content-Type': 'application/x-www-form-urlencoded'}
        self.token = requests.post("https://accounts.spotify.com/api/token",
                                   param, headers=header).\
            json()["access_token"]
        self.header = {"Authorization": "Bearer {}".format(self.token)}

    @rate_limited(200)
    def query(self, query, params=None):
        """
        Return a query string for the Spotify API.

        Args
            query: (str) URL coming after example.com
            params: (dict) of parameters specific to the qeury

        Return
            (json) Results from the Spotify API

        -----------------------------------------------------------------------
        """
        return requests.get(self.base_url + query, params,
                            headers=self.header).json()


class WasabiConnection(Connection):

    """Class object instantiates a connection with Wasabi."""

    def __init__(self):
        """Initialize Wasabi API connection."""
        Connection.__init__(self, "https://wasabi.i3s.unice.fr")


class AllMusicConnection(Connection):

    """Class object instantiates a connection with AllMusic.com."""

    def __init__(self, chromedriver_loc):
        """Initialize AllMusic.com connection."""
        Connection.__init__(self, "https://www.allmusic.com")
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless")
        self.browser = webdriver.Chrome(
            chromedriver_loc, chrome_options=chrome_options)

    @rate_limited(2)
    def query(self, query):
        """
        Execute an API query.

        query: String containing RESTful API query.
        """
        self.browser.get(self.base_url + query)


if __name__ == '__main__':

    conn = WasabiConnection()
    r = conn.query("/search/fulltext/Bon%20Jovi")
