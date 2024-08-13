# The purpose of this script is to fetch song data from Spotify API and save it as a CSV file.
import argparse

import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv


def get_playlists():
    return [
        '5IwFDvJvKVub47mVa4DPY0',  # angry
        '0sdnUIRzRN4Z3kvVI2wGA8',  # fear
        '7GhawGpb43Ctkq3PRP1fOL',  # happy
        '5DVUEqRL1EV8I9n65eBaAw',  # sad
    ]


def get_song_features():
    return [
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


# returns the track ids of the songs in the playlist
def get_songs_in_playlist(sp, playlist_id):
    results = sp.playlist_items(playlist_id, additional_types='tracks')
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return map(lambda x: x['track']['id'], tracks)


def get_features_of_track(sp, song_features, track_id):
    features = sp.audio_features(track_id)[0]
    return [float(features[key]) for key in song_features]


def create_dataframe(sp, playlists, song_features):
    columns = ['track_id']
    columns.extend(song_features)
    columns.append('emotion')
    df = pd.DataFrame(columns=columns)
    for emotion, playlist_id in enumerate(playlists):
        songs = get_songs_in_playlist(sp, playlist_id)
        for track_id in songs:
            row = [track_id]
            row.extend(get_features_of_track(sp, song_features, track_id))
            row.append(emotion)
            df.loc[len(df)] = row
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True, help='Output CSV path')
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()  # load Spotify API keys from .env file
    auth_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(auth_manager=auth_manager)

    playlists = get_playlists()
    song_features = get_song_features()
    df = create_dataframe(sp, playlists, song_features)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
