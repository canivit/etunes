# The purpose of this script is to fetch song data from Spotify API and save it as a CSV file.

import spotipy
import pandas as pd
import os
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# load Spotify API keys from .env file
load_dotenv()

playlists = [
    '5DVUEqRL1EV8I9n65eBaAw',  # sad
    '7GhawGpb43Ctkq3PRP1fOL',  # happy
    '5IwFDvJvKVub47mVa4DPY0',  # angry
    '0sdnUIRzRN4Z3kvVI2wGA8',  # fear
]

song_features = [
    'acousticness',  # [0.0 - 1.0]
    'danceability',  # [0.0 - 1.0]
    'energy',  # [0.0 - 1.0]
    'instrumentalness',  # [0.0 - 1.0]
    'liveness',  # [0.0 - 1.0]
    'loudness',  # [-60 - 0]
    'speechiness',  # [0.0 - 1.0]
    'tempo',  # API does not specify range
    'valence',  # [0.0 - 1.0]
]

auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)


# returns the track ids of the songs in the playlist
def get_songs_in_playlist(playlist_id):
    results = sp.playlist_items(playlist_id, additional_types='tracks')
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return map(lambda x: x['track']['id'], tracks)


def get_song_features(track_id):
    features = sp.audio_features(track_id)[0]
    return [float(features[key]) for key in song_features]


columns = song_features.copy()
columns.append('emotion')
df = pd.DataFrame(columns=columns)
for emotion, playlist_id in enumerate(playlists):
    songs = get_songs_in_playlist(playlist_id)
    for track_id in songs:
        features = get_song_features(track_id)
        features.append(emotion)
        df.loc[len(df)] = features

os.makedirs('data', exist_ok=True)
df.to_csv('data/songs.csv', index=False)
