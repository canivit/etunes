import torch
from torch import nn

from face.model import SimpleCNN

'''
Loads a checkpoint and saves it as a model file
'''


def main():
    checkpoint = torch.load('data/checkpoint_5.pt', map_location=torch.device('cpu'))
    model = nn.DataParallel(SimpleCNN())
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.module
    torch.save(model.state_dict(), 'data/model.pt')


if __name__ == "__main__":
    main()
