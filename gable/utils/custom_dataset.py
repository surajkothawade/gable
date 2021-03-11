
import numpy as np
import os
import torch
import torchvision
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import PIL.Image as Image
from sklearn.datasets import load_boston

#TODO: Add func for att imbalance

def create_class_imb(fullset, split_cfg, num_cls):
    train_idx = []
    val_idx = []
    lake_idx = []
    selected_classes = np.random.choice(np.arange(num_cls), size=split_cfg['num_cls_imbalance'], replace=False) #classes to imbalance
    for i in range(num_cls): #all_classes
        full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_imbclass_train'], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_lake'], replace=False))
        else:
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_class_train'], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_lake'], replace=False))
    
        train_idx += class_train_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx

    train_set = torch.utils.data.Subset(fullset, train_idx)
    val_set = torch.utils.data.Subset(fullset, val_idx)
    lake_set = torch.utils.data.Subset(fullset, lake_idx)
    return train_set, val_set, lake_set, selected_classes

def load_dataset_custom(datadir, dset_name, feature, split_cfg, isnumpy=False):
    if(not(os.path.exists(datadir))):
        os.mkdir(datadir)

    if(dset_name=="cifar10"):
        np.random.seed(42)
        num_cls=10
        cifar_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        fullset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=cifar_transform)
        test_set = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True, transform=cifar_transform)
        if(feature=="classimb"):
            train_set, val_set, lake_set = create_class_imb(fullset, split_cfg, num_cls)
        print("CIFAR-10 Custom dataset stats: Train size: ", len(train_idx), "Val size: ", len(val_idx), "Lake size: ", len(lake_idx))

        return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls

    if(dset_name=="mnist"):
        np.random.seed(42)
        num_cls=10
        mnist_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            transforms.Resize((32, 32)),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        fullset = torchvision.datasets.MNIST(root=datadir, train=True, download=True, transform=mnist_transform)
        test_set = torchvision.datasets.MNIST(root=datadir, train=False, download=True, transform=mnist_transform)
        if(feature=="classimb"):
            train_set, val_set, lake_set, imb_cls_idx = create_class_imb(fullset, split_cfg, num_cls)
        print("MNIST Custom dataset stats: Train size: ", len(train_idx), "Val size: ", len(val_idx), "Lake size: ", len(lake_idx))

        return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls

    if(dset_name=="cifar100"):
        np.random.seed(42)
        num_cls=100
        cifar100_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        fullset = torchvision.datasets.CIFAR100(root=datadir, train=True, download=True, transform=cifar100_transform)
        testset = torchvision.datasets.CIFAR100(root=datadir, train=False, download=True, transform=cifar100_transform)
        if(feature=="classimb"):
            train_set, val_set, lake_set = create_class_imb(fullset, split_cfg, num_cls)
        print("CIFAR-100 Custom dataset stats: Train size: ", len(train_idx), "Val size: ", len(val_idx), "Lake size: ", len(lake_idx))

        return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls

