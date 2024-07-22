import torch
import torch.nn as nn

from data_prep import prep_data
from model import EmotionCNN

model = EmotionCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load('model/best_model.pth'))
model.eval()

criterion = nn.CrossEntropyLoss()

test_loss = 0.0
correct = 0
total = 0

train_loader, val_loader, test_loader = prep_data()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_loader.dataset)
test_accuracy = 100 * correct / total

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
