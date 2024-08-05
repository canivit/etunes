# Load the trained model
import argparse
import os

import cv2
import torch
from torch import nn

from model import SimpleCNN, simple_cnn_transform

emotions = ['Angry', 'Fear', 'Happy', 'Sad', 'Neutral']
transform = simple_cnn_transform(augment=False)


def image_to_tensor(image):
    resized = cv2.resize(image, (48, 48)).astype('float32')
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    tensor = transform(grayscale)
    return tensor


def predict_emotion(model, device, tensor):
    tensor = tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        print(output)
        _, predicted = torch.max(output, 1)
        emotion_idx = predicted.item()
        return emotions[emotion_idx]


def load_checkpoint(file, model):
    if os.path.exists(file):
        checkpoint = torch.load(file, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Model is loaded from checkpoint file: {file}')
        return True
    else:
        return False


def run(model, device):
    # Capture the video from the webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect face using Haar Cascade
        haar_cascade_path = 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            tensor = image_to_tensor(face)
            emotion = predict_emotion(model, device, tensor)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--parallel', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    model = SimpleCNN()
    if args.parallel:
        model = nn.DataParallel(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model_loaded = load_checkpoint(args.checkpoint, model)
    if model_loaded:
        run(model, device)
    else:
        print(f'Failed to load model from file {args.checkpoint}')


if __name__ == "__main__":
    main()
