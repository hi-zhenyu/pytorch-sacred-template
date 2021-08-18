import numpy as np
import torch
import torchvision.datasets as dsets 
import torchvision.transforms as transforms
from torch.utils.data import Subset

def get_dataloader(data_dir, batch_size, split='train', val_split=None):
    if split == 'train':
        dataset = dsets.MNIST(root=data_dir,
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
        return train_dataloader, val_dataloader

    elif split == 'test':
        test_dataset = dsets.MNIST(root=data_dir,
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
        return test_dataloader
    else:
        raise ValueError('Invalid split: %s' % split)
