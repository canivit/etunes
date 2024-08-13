import argparse

import torch

from model import SongEmotionNetwork
from prep_data import get_song_features, get_playlists
from dataset import get_data_loaders


def evaluate(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels, track_ids in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    num_features = len(get_song_features())
    num_emotions = len(get_playlists())  # playlist for each emotion
    model = SongEmotionNetwork(input_dim=num_features, output_dim=num_emotions, hidden_dim=100)
    model.load_state_dict(torch.load(args.model))
    train_loader, val_loader, test_loader = get_data_loaders(args.data, args.batch_size)
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
