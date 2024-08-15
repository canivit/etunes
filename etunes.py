import argparse
import random

import cv2
import spotipy
import torch
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

from face.model import simple_cnn_transform, SimpleCNN
from music.dataset import get_data_loaders
from music.model import SongEmotionNetwork
from music.prep_data import get_song_features, get_playlists


def classify_songs(songs_csv, model, num_emotions, only_keep_correct_predictions):
    train_loader, val_loader, test_loader = get_data_loaders(songs_csv, 1)
    songs = [[] for _ in range(num_emotions)]
    model.eval()
    with torch.no_grad():
        for features, label, track_id in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.item()
            label = label.item()
            if only_keep_correct_predictions and predicted == label:
                songs[predicted].append((track_id[0], label))
            elif not only_keep_correct_predictions:
                songs[predicted].append((track_id[0], label))
    return songs


emotions = ['Angry', 'Fear', 'Happy', 'Sad', 'Neutral']
transform = simple_cnn_transform(augment=False)


def image_to_tensor(image):
    resized = cv2.resize(image, (48, 48)).astype('float32')
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    tensor = transform(grayscale)
    return tensor


def predict_emotion(model, tensor):
    tensor = tensor.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()


def play_song(songs, predicted_emotion_idx, sp_client):
    if predicted_emotion_idx == 4:
        return

    track_id, true_emotion_idx = random.choice(songs[predicted_emotion_idx])
    predicted_emotion = emotions[predicted_emotion_idx]
    true_emotion = emotions[true_emotion_idx]
    print(f'Playing song {track_id}')
    print(f'Predicted: {predicted_emotion}, Actual: {true_emotion}')
    sp_client.start_playback(uris=[f'spotify:track:{track_id}'])
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            break

    playback = sp_client.current_playback()
    if playback and playback['is_playing']:
        sp_client.pause_playback()


def setup_capture_device():
    # Capture the video from the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set the width to 1280 pixels
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set the height to 720 pixels
    return cap


def capture_face(capture_device, cascade):
    ret, frame = capture_device.read()
    # Detect face using Haar Cascade
    face_cascade = cv2.CascadeClassifier(cascade)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return frame, None
    else:
        return frame, faces[0]


def draw_face_and_emotion(frame, face_x, face_y, face_w, face_h, emotion):
    cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
    cv2.putText(frame, emotion, (face_x, face_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)


def run(face_model, songs, sp_client, cascade):
    cap = setup_capture_device()
    while True:
        frame, face = capture_face(cap, cascade)
        if face is None:
            continue

        (face_x, face_y, face_w, face_h) = face
        tensor = image_to_tensor(frame[face_y:face_y + face_h, face_x:face_x + face_w])
        emotion_idx = predict_emotion(face_model, tensor)
        emotion = emotions[emotion_idx]
        draw_face_and_emotion(frame, face_x, face_y, face_w, face_h, emotion)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            play_song(songs, emotion_idx, sp_client)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--redirect-uri', type=str, default='http://localhost:8080',
                        help='Spotify Authentication URL')
    parser.add_argument('--face-model', type=str, default='face/data/model.pt')
    parser.add_argument('--cascade', type=str, default='face/haarcascade_frontalface_default.xml')
    parser.add_argument('--song-model', type=str, default='music/data/model.pt')
    parser.add_argument('--song-data', type=str, default='music/data/songs.csv')
    parser.add_argument('--always-correct-song', action='store_true')
    return parser.parse_args()


def setup_spotify_client(redirect_uri):
    load_dotenv()  # load Spotify API keys from .env file
    auth_manager = SpotifyOAuth(scope='user-modify-playback-state user-read-playback-state',
                                redirect_uri=redirect_uri)
    sp_client = spotipy.Spotify(auth_manager=auth_manager)
    return sp_client


def main():
    args = parse_args()
    sp_client = setup_spotify_client(args.redirect_uri)

    num_features = len(get_song_features())
    num_emotions = len(get_playlists())  # playlist for each emotion
    music_model = SongEmotionNetwork(input_dim=num_features, output_dim=num_emotions, hidden_dim=100)
    music_model.load_state_dict(torch.load(args.song_model))
    songs = classify_songs(args.song_data, music_model, num_emotions, args.always_correct_song)

    face_model = SimpleCNN()
    face_model.load_state_dict(torch.load(args.face_model))

    run(face_model, songs, sp_client, args.cascade)


if __name__ == "__main__":
    main()
