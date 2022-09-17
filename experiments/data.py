from torch.utils import data
import torch
import pickle

class ArtificialStructuresDataset(data.Dataset):
    def __init__(self, data_path, device):
        with open(data_path,'rb') as f:
            # send data to GPU if possible
            self.data = pickle.load(f).to(device)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def noise_batch(batch_size, noise_dim, device="cpu", normal_dist=True):
    # TODO: add more random distributions
    if normal_dist:
        return torch.Tensor(batch_size, noise_dim).normal_(0, 0.33).to(device)
    return torch.randn(batch_size, noise_dim).to(device)
