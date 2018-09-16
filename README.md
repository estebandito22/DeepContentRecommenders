# DeepContentRecommenders
### Implements multiple methods for content based music recommendations using deep learning.  

The two approaches included are developed by van den Oord et. al. [2013] and Lee et. al. [2018].
In contrast to collaborative filtering based methods, the content based methods implemented here use audio from songs to develop recommendations for users.
First, the audio is transformed into a spectrogram.
Both models use the spectrogram as input to a convolutional neural network to either 1) predict the song embeddings generated from a Weighted Regularized Matrix Factorization model, or 2) Directly embed the songs into a low dimensional representation that can be used to make recommendations to users.


### Repository Structure:

/data - contains classes for downloading songs from the WasabiAPI and aligning the metadata of those songs with the Million Song Dataset.

/dc - contains both the DCBR (van den Oord et. al [2013]) and DCUE (Lee et. al. [2018]) implementations as well as PyTorch data sets that are used for training these models.

/dc/dcbr - contains classes for the collaborative filtering based model and neural network that are used to construct the DCBR model.

/dc/dcue - contains classes to construct the DCUE model.

/dc/nn - contains classes for training both the DCBR and DCUE models.

train_*.py - are scripts that use the trainer classes to train the models.

eval_*.py - are scripts that use the previously trained models and evaluate them on the test sets.

download_audio.py - collects Million Song Dataset audio and metadata from the WasabiAPI.

transform_audio.py - preprocesses the raw audio into PyTorch tensors containing the spectrograms.
