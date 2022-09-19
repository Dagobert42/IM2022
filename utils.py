import torch
from torch.utils import data
import torch.autograd as autograd
from torch.autograd import Variable
import pickle
import numpy as np
from numpy.random import rand

class ArtificialStructuresDataset(data.Dataset):
    def __init__(self, data_path, device):
        with open(data_path,'rb') as f:
            # send data to GPU if possible
            self.data = pickle.load(f).to(device)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# # TODO: cite
# def W_div(samples, critique_samples, fakes, critique_fakes):
#     K = 2
#     P = 6
#     real_grad_out = Variable(Tensor(samples.size(0), 1).fill_(1.0), requires_grad=False)
#     real_grad = autograd.grad(
#         critique_samples, samples, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
#     )[0]
#     real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (P / 2)

#     fake_grad_out = Variable(Tensor(fakes.size(0), 1).fill_(1.0), requires_grad=False)
#     fake_grad = autograd.grad(
#         critique_fakes, fakes, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
#     )[0]
#     fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (P / 2)

#     div_gp = torch.mean(real_grad_norm + fake_grad_norm) * K / 2

#     # Adversarial loss
#     D_loss = -torch.mean(critique_samples) + torch.mean(critique_fakes) + div_gp
#     return D_loss