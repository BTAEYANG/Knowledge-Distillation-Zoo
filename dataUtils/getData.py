import os.path

import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler, DataLoader


# Get Data Loader
def getDataLoader(root_path: str = '/home/lab265/lab265/datasets/', split_factor: float = 0.1, seed: int = 66,
                  data_set: str = 'CIFAR10'):
    data_set_path = os.path.join(root_path, data_set)

    if data_set == 'CIFAR10':
        print('=> loading cifar10 data...')
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

        train_Transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_Transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=data_set_path, train=True, download=True, transform=train_Transforms)

        test_set = torchvision.datasets.CIFAR10(
            root=data_set_path, train=False, download=True, transform=test_Transforms)

    elif data_set == 'CIFAR100':

        print('=> loading cifar100 data...')
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_Transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])

        test_Transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_set = torchvision.datasets.CIFAR100(
            root=data_set_path, train=True, download=True, transform=train_Transforms)

        test_set = torchvision.datasets.CIFAR100(
            root=data_set_path, train=False, download=True, transform=test_Transforms)

    dataset_size = len(train_set)
    indices = list(range(dataset_size))
    split = int(np.floor(split_factor * dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_set, batch_size=128, sampler=train_sampler,
                              num_workers=4, drop_last=False, pin_memory=True)
    validation_loader = DataLoader(train_set, batch_size=100, sampler=valid_sampler,
                                   num_workers=4, drop_last=False,
                                   pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)
    return train_loader, validation_loader, test_loader
