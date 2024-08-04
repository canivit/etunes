import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn

from fer2013 import get_data_loaders
from model import SimpleCNN, simple_cnn_transform


def train_epoch(model, optimizer, train_loader, loss_fn, device, epoch, start_batch_idx, dst_checkpoint):
    print(f'Starting epoch {epoch}')
    model.train()
    total_loss = 0
    count = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx < start_batch_idx:
            continue

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
            save_checkpoint(model, optimizer, epoch, batch_idx + 1, dst_checkpoint)

    avg_loss = total_loss / count
    print(f'Completed epoch {epoch}, Loss: {avg_loss:.4f}')
    save_checkpoint(model, optimizer, epoch + 1, 0, dst_checkpoint)


def train(model, optimizer, train_loader, loss_fn, device, start_epoch, end_epoch, start_batch,
          dst_checkpoint):
    for epoch in range(start_epoch, end_epoch):
        if epoch != start_epoch:
            start_batch = 0
        train_epoch(model, optimizer, train_loader, loss_fn, device, epoch, start_batch, dst_checkpoint)
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
    if checkpoint_path is None:
        print(f'No checkpoint is provided. Starting training from scratch.')
        return 0, 0
    elif os.path.exists(checkpoint_path):
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


def is_sagemaker():
    return 'TRAINING_JOB_NAME' in os.environ


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Number of batches between checkpoints')
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--src-checkpoint', type=str, required=False)
    parser.add_argument('--dst-checkpoint', type=str, required=True)
    args = parser.parse_args()

    data_file = args.data_file
    src_checkpoint = args.src_checkpoint
    dst_checkpoint = args.dst_checkpoint
    if is_sagemaker():
        data_dir = os.environ["SM_CHANNEL_TRAIN"]
        checkpoint_dir = '/opt/ml/checkpoints'
        data_file = os.path.join(data_dir, data_file)
        dst_checkpoint = os.path.join(checkpoint_dir, dst_checkpoint)
        if src_checkpoint is not None:
            src_checkpoint = os.path.join(checkpoint_dir, src_checkpoint)

    model = SimpleCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    device = 'cpu'
    if torch.cuda.is_available():
        print(f'Cuda is available. Using {torch.cuda.device_count()} GPU(s)')
        device = 'cuda'
        cudnn.benchmark = True
        model = nn.DataParallel(model)

    model.to(device)
    transform = simple_cnn_transform()
    train_loader, test_loader = get_data_loaders(data_file, args.batch_size, args.num_workers, transform)
    start_epoch, start_batch = load_checkpoint(src_checkpoint, model, optimizer)
    train(model, optimizer, train_loader, loss_fn, device, start_epoch, args.epochs, start_batch, dst_checkpoint)
