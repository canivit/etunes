import argparse
import os
import torch
import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn

from fer2013 import get_data_loaders
from model import SimpleCNN, simple_cnn_transform


def eval_loop(model, test_loader, device):
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for idx, (images, labels) in enumerate(tqdm.tqdm(test_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())

        accuracy = correct / total * 100
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f'Accuracy: {accuracy:.2f}%')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')


def load_checkpoint(file, model):
    if os.path.exists(file):
        checkpoint = torch.load(file, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Model is loaded from checkpoint file: {file}')
        return True
    else:
        return False


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--data-file', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--parallel', action='store_true')
args = parser.parse_args()

model = SimpleCNN()
if args.parallel:
    model = nn.DataParallel(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

transform = simple_cnn_transform(augment=False)
train_loader, val_loader, test_loader = get_data_loaders('data/fer2013.csv', args.batch_size, 0, transform)
model_loaded = load_checkpoint(args.checkpoint, model)
if model_loaded:
    eval_loop(model, test_loader, device)
else:
    print(f'Failed to load model from file {args.checkpoint}')
