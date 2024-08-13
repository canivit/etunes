import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class SongDataset(Dataset):
    def __init__(self, features, labels, track_ids):
        self.features = features
        self.labels = labels
        self.track_ids = track_ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.track_ids[idx]


def get_features_and_labels(df):
    # split data into features, labels, and track_ids
    track_ids = df.iloc[:, 0].to_numpy(dtype=str)
    x = df.iloc[:, 1:-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)

    # normalize features
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    x = x.astype(np.float32)

    # convert to tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return x, y, track_ids


def create_datasets(csv_file):
    df = pd.read_csv(csv_file)

    # split dataset into train, validation, and test
    traindf, tempdf = train_test_split(df, test_size=0.3, random_state=6014, stratify=df['emotion'])
    valdf, testdf = train_test_split(tempdf, test_size=0.5, random_state=6014, stratify=tempdf['emotion'])

    x_train, y_train, tracks_train = get_features_and_labels(traindf)
    x_val, y_val, tracks_val = get_features_and_labels(valdf)
    x_test, y_test, tracks_test = get_features_and_labels(testdf)

    # create datasets
    train_dataset = SongDataset(x_train, y_train, tracks_train)
    val_dataset = SongDataset(x_val, y_val, tracks_val)
    test_dataset = SongDataset(x_test, y_test, tracks_test)

    return train_dataset, val_dataset, test_dataset


def get_data_loaders(csv_file, batch_size):
    train_dataset, val_dataset, test_dataset = create_datasets(csv_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
