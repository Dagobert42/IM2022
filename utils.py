import torch
from torch.utils import data
import pickle
from numpy.random import rand
from enum import Enum

# device defintions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class ModelType(Enum):
    DCGAN = "Deep Conv"
    WGANdiv = "WGAN div"

class ArtificialStructuresDataset(data.Dataset):
    def __init__(self, data_path, device):
        with open(data_path,'rb') as f:
            # send data to GPU if possible
            self.data = pickle.load(f).to(device)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data[:64])
