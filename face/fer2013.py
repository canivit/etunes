import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


def prep_data(csv_file):
    # Load the dataset
    data = pd.read_csv(csv_file)
    emotion_idx_map = {0: 0, 2: 1, 3: 2, 4: 3, 6: 4}
    # emotion_map = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Neutral'}

    # remove `disgust` and `surprise` emotions
    df = data[(data['emotion'] != 1) & (data['emotion'] != 5)].copy()
    # change emotion indices
    df['emotion'] = df['emotion'].apply(lambda x: emotion_idx_map[x])
    # change pixels column to numpy array
    df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))

    # split train and test data
    x_train = np.array(df[df['Usage'] == 'Training']['pixels'].tolist())
    y_train = np.array(df[df['Usage'] == 'Training']['emotion'].tolist())

    x_val = np.array(df[df['Usage'] == 'PrivateTest']['pixels'].tolist())
    y_val = np.array(df[df['Usage'] == 'PrivateTest']['emotion'].tolist())

    x_test = np.array(df[df['Usage'] == 'PublicTest']['pixels'].tolist())
    y_test = np.array(df[df['Usage'] == 'PublicTest']['emotion'].tolist())

    # reshape pixel arrays
    x_train = x_train.reshape(-1, 48, 48)
    x_val = x_val.reshape(-1, 48, 48)
    x_test = x_test.reshape(-1, 48, 48)

    assert len(x_train) == len(y_train)
    assert len(x_val) == len(y_val)
    assert len(x_test) == len(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


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


def get_data_loaders(csv_file, batch_size, num_workers, transform=None):
    x_train, y_train, x_val, y_val, x_test, y_test = prep_data(csv_file)

    train_dataset = FER2013Dataset(x_train, y_train, transform=transform)
    val_dataset = FER2013Dataset(x_val, y_val, transform=transform)
    test_dataset = FER2013Dataset(x_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
