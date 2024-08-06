import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class SongDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def get_features_and_labels(csv_file):
    data = pd.read_csv(csv_file)

    # split data into features and labels
    x = data.iloc[:, 1:-1].values.astype(np.float32)
    y = data.iloc[:, -1].values.astype(np.float32)

    # normalize features
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    x = x.astype(np.float32)

    return x, y


def create_datasets(x, y):
    # split dataset into train, validation, and test
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=6014)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=6014)

    # convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # create datasets
    train_dataset = SongDataset(x_train, y_train)
    val_dataset = SongDataset(x_val, y_val)
    test_dataset = SongDataset(x_test, y_test)

    return train_dataset, val_dataset, test_dataset


def get_data_loaders(csv_file, batch_size):
    x, y = get_features_and_labels(csv_file)
    train_dataset, val_dataset, test_dataset = create_datasets(x, y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
