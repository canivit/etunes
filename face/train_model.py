import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from fer2013 import prep_data, get_data_loaders
from model import SimpleCNN, SimpleVGG


def train_epoch(model, optimizer, train_loader, loss_fn, device, curr_epoch, start_batch):
    model.train()
    total_loss = 0
    count = 0
    for batch_idx, (images, labels) in enumerate(tqdm.tqdm(train_loader), start=start_batch):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1
        if batch_idx % 10 == 0:
            save_checkpoint(model, optimizer, curr_epoch, batch_idx + 1, 'data/vgg_checkpoint.pt')

    avg_loss = total_loss / count
    print(f'Epoch {curr_epoch}, Loss: {avg_loss:.4f}')
    torch.save(model.state_dict(), 'model.pth')


def train(model, optimizer, train_loader, loss_fn, device, start_epoch, end_epoch, start_batch):
    for epoch in range(start_epoch, end_epoch):
        if epoch != start_epoch:
            start_batch = 0
        train_epoch(model, optimizer, train_loader, loss_fn, device, epoch, start_batch)
        if epoch > start_epoch:
            continue


def save_checkpoint(model, optimizer, epoch, batch, file):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(checkpoint, file)


def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch']

        print(f'Checkpoint loaded: {checkpoint_path}')
        print(f'Resuming from epoch {start_epoch}, batch {start_batch}')

        return start_epoch, start_batch
    else:
        print(f'No checkpoint found at {checkpoint_path}. Starting training from scratch.')
        return 0, 0


model = SimpleVGG()
learning_rate = 0.001
num_epochs = 100
batch_size = 64
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
train_loader, test_loader = get_data_loaders(batch_size)
start_epoch, start_batch = load_checkpoint('data/vgg_checkpoint.pt', model, optimizer)
train(model, optimizer, train_loader, loss_fn, device, start_epoch, num_epochs, start_batch)
