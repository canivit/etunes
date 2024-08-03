import os
import torch
import tqdm

from fer2013 import get_data_loaders, vgg_transform
from model import SimpleCNN, create_custom_vgg


def eval_loop(model, test_loader, device):
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

        print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')


def load_checkpoint(file, model):
    if os.path.exists(file):
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Model is loaded from checkpoint file: {file}')
        return True
    else:
        return False


batch_size = 64
checkpoint_file = 'data/vgg_checkpoint.pt'
model = create_custom_vgg(num_classes=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
transform = vgg_transform()
train_loader, test_loader = get_data_loaders('data/fer2013.csv', batch_size, transform)
model_loaded = load_checkpoint(checkpoint_file, model)
if model_loaded:
    eval_loop(model, test_loader, device)
else:
    print(f'Failed to load model from file {checkpoint_file}')
