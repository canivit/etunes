import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_data_loaders
from model import SongEmotionNetwork
from prep_data import get_song_features, get_playlists


def train_epoch(model, optimizer, loss_fn, train_loader):
    model.train()
    total_loss = 0
    for features, labels, track_ids in train_loader:
        # Forward pass
        outputs = model(features)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, loss_fn, val_loader):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for features, labels, track_ids in val_loader:
            # Forward pass
            outputs = model(features)
            loss = loss_fn(outputs, labels)

            # Update loss
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train(model, optimizer, loss_fn, num_epochs, train_loader, val_loader):
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, optimizer, loss_fn, train_loader)
        val_loss, val_accuracy = validate(model, loss_fn, val_loader)
        print(f'Completed epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Acc: {val_accuracy}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    num_features = len(get_song_features())
    num_emotions = len(get_playlists())  # playlist for each emotion
    model = SongEmotionNetwork(input_dim=num_features, output_dim=num_emotions, hidden_dim=100)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_loader, val_loader, test_loader = get_data_loaders(args.data, args.batch_size)
    train(model, optimizer, loss_fn, args.epochs, train_loader, val_loader)
    torch.save(model.state_dict(), args.model)


if __name__ == "__main__":
    main()
