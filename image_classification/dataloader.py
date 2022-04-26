import torch
from torch.utils.data.sampler import SubsetRandomSampler as sps
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import numpy as np


def getdataloader(args):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])
    
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False, download=True, transform=transform)
    indices = np.arange(len(train_data))
    train_indeces = np.random.choice(indices, int(0.8*len(indices)), replace=False)
    test_indeces = np.array(list(set(indices) - set(train_indeces)))
    train_sampler = sps(train_indeces)
    valid_sampler = sps(test_indeces)
    
    train_loader = DataLoader(train_data, batch_size = args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_data, batch_size = args.batch_size, sampler=valid_sampler)
    test_loader = DataLoader(test_data, batch_size = args.batch_size)
    return train_loader, val_loader, test_loader