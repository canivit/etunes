import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# load data
data = pd.read_csv('data/songs.csv')
data

# split data into features and labels
x = data.iloc[:, 0:-1].values.astype(np.float32)
y = data.iloc[:, -1].values.astype(np.float32)

# normalize features
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x = x.astype(np.float32)

# split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6014)
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5, random_state=6014)

# convert to tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
x_validation = torch.tensor(x_validation, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
y_validation = torch.tensor(y_validation, dtype=torch.long)


# create training and testing datasets
class SongDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = SongDataset(x_train, y_train)
test_dataset = SongDataset(x_test, y_test)
validation_dataset = SongDataset(x_validation, y_validation)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
