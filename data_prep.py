import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FER2013Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def prep_data():
    # Load the dataset
    data = pd.read_csv('data/fer2013.csv')
    # Convert pixels column to numpy array
    data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    # Create feature and label arrays
    x = np.array(data['pixels'].tolist())
    y = np.array(data['emotion'].tolist())
    # Reshape X to match the input shape of the CNN
    x = x.reshape(-1, 48, 48, 1)
    # Normalize the pixel values
    x /= 255.0
    # Split the data
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = FER2013Dataset(x_train, y_train, transform=transform)
    val_dataset = FER2013Dataset(x_val, y_val, transform=transform)
    test_dataset = FER2013Dataset(x_test, y_test, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader
