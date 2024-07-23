# The purpose of this script is to fetch song data from Spotify API and save it as a CSV file.

import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# load Spotify API keys from .env file
load_dotenv()

playlists = [
    '5DVUEqRL1EV8I9n65eBaAw',  # sad
    # '',  # happy
    # '',  # angry
    # '',  # fear
]

song_features = [
    'acousticness',
    'danceability',
    'energy',
    'instrumentalness',
    'liveness',
    'loudness',
    'speechiness',
    'tempo',
    'valence',
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
    return [features[key] for key in song_features]


columns = song_features.copy()
columns.append('emotion')
df = pd.DataFrame(columns=columns)
for emotion, playlist_id in enumerate(playlists):
    songs = get_songs_in_playlist(playlist_id)
    for track_id in songs:
        features = get_song_features(track_id)
        features.append(emotion)
        df.loc[len(df)] = features

print(df)