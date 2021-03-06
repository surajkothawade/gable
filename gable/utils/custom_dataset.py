
import numpy as np
import os
import torch
import torchvision
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
import PIL.Image as Image
from gable.gable.utils.custom_utils import *
from sklearn.datasets import load_boston
np.random.seed(42)
torch.manual_seed(42)

# class custom_subset(Dataset):
#     r"""
#     Subset of a dataset at specified indices.

#     Arguments:
#         dataset (Dataset): The whole Dataset
#         indices (sequence): Indices in the whole set selected for subset
#         labels(sequence) : targets as required for the indices. will be the same length as indices
#     """
#     def __init__(self, dataset, indices, labels):
#         # self.dataset = torch.repeat_interleave(dataset.data[indices].unsqueeze(1), 3, 1)
#         self.dataset = dataset[indices]
#         self.targets = labels.type(torch.long)
#     def __getitem__(self, idx):
#         image = self.dataset[idx]

#         target = self.targets[idx]
#         return (image, target)

#     def __len__(self):
#         return len(self.targets)

class DataHandler_MNIST(Dataset):
    """
    Data Handler to load MNIST dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    """

    def __init__(self, X, Y=None, select=True, use_test_transform=False):
        """
        Constructor
        """
        self.select = select
        self.use_test_transform=use_test_transform
        self.training_gen_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.test_gen_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if not self.select:
            self.X = X
            self.targets = Y
        else:
            self.X = X

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.targets[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            if(x.shape[0]==1): x = torch.repeat_interleave(x, 3, 0)
            y=y.long()
            return (x, y.long())

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            if(x.shape[0]==1): x = torch.repeat_interleave(x, 3, 0)
            return x

    def __len__(self):
        return len(self.X)

class DataHandler_CIFAR10(Dataset):
    """
    Data Handler to load CIFAR10 dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    """

    def __init__(self, X, Y=None, select=True, use_test_transform = False):
        """
        Constructor
        """
        self.select = select
        if(use_test_transform):
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        if not self.select:
            self.X = X
            self.targets = Y
            self.transform = transform
        else:
            self.X = X
            self.transform = transform

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.targets[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return (x, y)

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return x

    def __len__(self):
        return len(self.X)

class DataHandler_SVHN(Dataset):
    """
    Data Handler to load SVHN dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    """

    def __init__(self, X, Y=None, select=True, use_test_transform=False):
        """
        Constructor
        """
        self.select = select
        self.use_test_transform=use_test_transform
        self.training_gen_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.test_gen_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std
        if not self.select:
            self.X = X
            self.targets = Y
        else:
            self.X = X

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.targets[index]
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            return (x, y)

        else:
            x = self.X[index]
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            return x

    def __len__(self):
        return len(self.X)

class DataHandler_UTKFace(Dataset):
    """
    Data Handler to load UTKFace dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    """
    def __init__(self, X, Y=None, select=True, use_test_transform = False):
        """
        Constructor
        """
        self.select = select
        if(use_test_transform):
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std
        else:
            transform = transforms.Compose([transforms.RandomCrop(200, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std
        if not self.select:
            self.X = X
            self.targets = Y
            self.transform = transform
        else:
            self.X = X
            self.transform = transform

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.targets[index]
            x = Image.fromarray(np.transpose(x, (1,2,0)))
            x = self.transform(x)
            return (x, y)

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return x

    def __len__(self):
        return len(self.X)
    
class DuplicateChannels(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return torch.repeat_interleave(pic.unsqueeze(1), 3, 1).float()

    def __repr__(self):
        return self.__class__.__name__ + '()'

def getOODtargets(targets, sel_cls_idx, ood_cls_id):
    ood_targets = []
    targets_list = list(targets)
    for i in range(len(targets_list)):
        if(targets_list[i] in list(sel_cls_idx)):
            ood_targets.append(targets_list[i])
        else:
            ood_targets.append(ood_cls_id)
    print("num ood samples: ", ood_targets.count(ood_cls_id))
    return torch.Tensor(ood_targets)
    
def create_ood_data(dset_name, fullset, testset, split_cfg, num_cls, isnumpy, augVal):
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    test_idx = []
    # selected_classes = np.random.choice(np.arange(num_cls), size=split_cfg['num_cls_idc'], replace=False) #number of in distribution classes
    # if(dset_name=="mnist"):
    #     selected_classes = np.array(list(range(num_cls)[num_cls-split_cfg['num_cls_idc']:]))
    # else:    
    selected_classes = np.array(list(range(split_cfg['num_cls_idc'])))
    for i in range(num_cls): #all_classes
        if(dset_name=="mnist"):
            full_idx_class = list(torch.where(torch.Tensor(fullset.targets.float()) == i)[0].cpu().numpy())
        else:    
            full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
            if(dset_name=="mnist"):
                test_idx_class = list(torch.where(torch.Tensor(testset.targets.float()) == i)[0].cpu().numpy())
            else:
                test_idx_class = list(torch.where(torch.Tensor(testset.targets) == i)[0].cpu().numpy())
            test_idx += test_idx_class
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_idc_train'], replace=False))
            train_idx += class_train_idx
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_idc_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_idc_lake'], replace=False))
        else:
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_ood_train'], replace=False)) #always 0
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_ood_val'], replace=False)) #Only for CG ood val has samples
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_ood_lake'], replace=False)) #many ood samples in lake
    
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
    if(dset_name=="mnist"):
        train_set = custom_mnist_subset(fullset, train_idx, torch.Tensor(fullset.targets.float())[train_idx])
        val_set = custom_mnist_subset(fullset, val_idx, torch.Tensor(fullset.targets.float())[val_idx])
        lake_set = custom_mnist_subset(fullset, lake_idx, getOODtargets(torch.Tensor(fullset.targets.float())[lake_idx], selected_classes, split_cfg['num_cls_idc']))
        test_set = custom_mnist_subset(testset, test_idx, torch.Tensor(testset.targets.float())[test_idx])
    else:
        train_set = custom_subset(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
        val_set = custom_subset(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
        lake_set = custom_subset(fullset, lake_idx, getOODtargets(torch.Tensor(fullset.targets)[lake_idx], selected_classes, split_cfg['num_cls_idc']))
        test_set = custom_subset(testset, test_idx, torch.Tensor(testset.targets)[test_idx])
    if(isnumpy):        
        if(dset_name=="mnist"):
            # X  = np.resize(fullset.data.float().cpu().numpy(), (len(fullset),32,32))
            X  = fullset.data.numpy()
            y = torch.from_numpy(np.array(fullset.targets.float()))
        else:            
            X = fullset.data
            y = torch.from_numpy(np.array(fullset.targets))
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_unlabeled = X[lake_idx]
        y_unlabeled = y[lake_idx]
        X_test = testset.data[test_idx]
        y_test = torch.from_numpy(np.array(testset.targets))[test_idx]
        return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, selected_classes
    else:
        return train_set, val_set, test_set, lake_set, selected_classes

def create_class_imb(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal):
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    if(dset_name=="mnist"): selected_classes=np.array([5,8])
    else: selected_classes = np.random.choice(np.arange(num_cls), size=split_cfg['num_cls_imbalance'], replace=False) #classes to imbalance
    for i in range(num_cls): #all_classes
        if(dset_name=="mnist"):
            full_idx_class = list(torch.where(torch.Tensor(fullset.targets.float()) == i)[0].cpu().numpy())
        elif(dset_name=="svhn"):
            full_idx_class = list(torch.where(torch.Tensor(fullset.labels) == i)[0].cpu().numpy())
        else:    
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
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
    # if(dset_name=="mnist"):
    #     train_set = custom_subset(torch.repeat_interleave(fullset.data.float().unsqueeze(1), 3, 1), train_idx, torch.Tensor(fullset.targets.float())[train_idx])
    #     val_set = custom_subset(torch.repeat_interleave(fullset.data.float().unsqueeze(1), 3, 1), val_idx, torch.Tensor(fullset.targets.float())[val_idx])
    #     lake_set = custom_subset(torch.repeat_interleave(fullset.data.float().unsqueeze(1), 3, 1), lake_idx, torch.Tensor(fullset.targets.float())[lake_idx])
    # else:
    if(dset_name=="mnist"):
        train_set = custom_mnist_subset(fullset, train_idx, torch.Tensor(fullset.targets.float())[train_idx])
        val_set = custom_mnist_subset(fullset, val_idx, torch.Tensor(fullset.targets.float())[val_idx])
        lake_set = custom_mnist_subset(fullset, lake_idx, torch.Tensor(fullset.targets.float())[lake_idx])
    elif(dset_name=="svhn"):
        train_set = custom_subset(fullset, train_idx, torch.Tensor(fullset.labels)[train_idx])
        val_set = custom_subset(fullset, val_idx, torch.Tensor(fullset.labels)[val_idx])
        lake_set = custom_subset(fullset, lake_idx, torch.Tensor(fullset.labels)[lake_idx])        
    else:
        train_set = custom_subset(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
        val_set = custom_subset(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
        lake_set = custom_subset(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    
    if(isnumpy):
        if(dset_name=="mnist"):
            # X = torch.repeat_interleave(fullset.data.float().unsqueeze(1), 3, 1).numpy()
            # X  = np.resize(fullset.data.float().cpu().numpy(), (len(fullset),32,32))
            X  = fullset.data.numpy()
            y = torch.from_numpy(np.array(fullset.targets.float()))
        elif(dset_name=="svhn"):
            X = fullset.data
            y = torch.from_numpy(np.array(fullset.labels))
        else:            
            X = fullset.data
            y = torch.from_numpy(np.array(fullset.targets))
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_unlabeled = X[lake_idx]
        y_unlabeled = y[lake_idx]
        return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set, selected_classes
    else:
        return train_set, val_set, lake_set, selected_classes

def getDuplicateData(dset_name, fullset, split_cfg):
    num_rep=split_cfg['num_rep']
    if(dset_name=="mnist"):
        # X  = np.resize(fullset.data.float().cpu().numpy(), (len(fullset),32,32))
        X  = fullset.data.numpy()
        y = torch.from_numpy(np.array(fullset.targets.float()))
    elif(dset_name == "svhn"):
        X = fullset.data
        y = torch.from_numpy(np.array(fullset.labels))
    else:
        X = fullset.data
        y = torch.from_numpy(np.array(fullset.targets))
    X_tr = X[:split_cfg['train_size']]
    y_tr = y[:split_cfg['train_size']]
    X_unlabeled = X[split_cfg['train_size']:len(X)-split_cfg['val_size']]
    y_unlabeled = y[split_cfg['train_size']:len(X)-split_cfg['val_size']]
    X_val = X[len(X)-split_cfg['val_size']:]
    y_val = y[len(X)-split_cfg['val_size']:]
    X_unlabeled_rep = np.repeat(X_unlabeled[:split_cfg['lake_subset_repeat_size']], num_rep, axis=0)
    y_unlabeled_rep = np.repeat(y_unlabeled[:split_cfg['lake_subset_repeat_size']], num_rep, axis=0)
    assert((X_unlabeled_rep[0]==X_unlabeled_rep[num_rep-1]).all())
    assert((y_unlabeled_rep[0]==y_unlabeled_rep[num_rep-1]).all())
    X_unlabeled_rep = np.concatenate((X_unlabeled_rep, X_unlabeled[split_cfg['lake_subset_repeat_size']:split_cfg['lake_size']]), axis=0)
    y_unlabeled_rep = torch.from_numpy(np.concatenate((y_unlabeled_rep, y_unlabeled[split_cfg['lake_subset_repeat_size']:split_cfg['lake_size']]), axis=0))
    if(dset_name=="mnist"):
        train_set = DataHandler_MNIST(X_tr, y_tr, False)
        lake_set = DataHandler_MNIST(X_unlabeled_rep, y_unlabeled_rep, False)
        val_set = DataHandler_MNIST(X_val, y_val, False)
    elif(dset_name=="svhn"):
        train_set = DataHandler_SVHN(X_tr, y_tr, False)
        lake_set = DataHandler_SVHN(X_unlabeled_rep, y_unlabeled_rep, False)
        val_set = DataHandler_SVHN(X_val, y_val, False)      
    else:
        train_set = DataHandler_CIFAR10(X_tr, y_tr, False)
        lake_set = DataHandler_CIFAR10(X_unlabeled_rep, y_unlabeled_rep, False)
        val_set = DataHandler_CIFAR10(X_val, y_val, False)
    return X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, lake_set

def getVanillaData(dset_name, fullset, split_cfg):
    if(dset_name=="mnist"):
        # X  = np.resize(fullset.data.float().cpu().numpy(), (len(fullset),32,32))
        X  = fullset.data.numpy()
        y = torch.from_numpy(np.array(fullset.targets.float()))
    else:
        X = fullset.data
        y = torch.from_numpy(np.array(fullset.targets))
    X_tr = X[:split_cfg['train_size']]
    y_tr = y[:split_cfg['train_size']]
    X_unlabeled = X[split_cfg['train_size']:len(X)-split_cfg['val_size']]
    y_unlabeled = y[split_cfg['train_size']:len(X)-split_cfg['val_size']]
    X_val = X[len(X)-split_cfg['val_size']:]
    y_val = y[len(X)-split_cfg['val_size']:]
    if(dset_name=="mnist"):
        train_set = DataHandler_MNIST(X_tr, y_tr, False)
        lake_set = DataHandler_MNIST(X_unlabeled, y_unlabeled, False)
        val_set = DataHandler_MNIST(X_val, y_val, False)
    else:
        train_set = DataHandler_CIFAR10(X_tr, y_tr, False)
        lake_set = DataHandler_CIFAR10(X_unlabeled, y_unlabeled, False)
        val_set = DataHandler_CIFAR10(X_val, y_val, False)
    return X_tr, y_tr, X_val, y_val, X_unlabeled[:split_cfg['lake_size']], y_unlabeled[:split_cfg['lake_size']], train_set, val_set, lake_set

def create_class_imb_bio(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal):
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    selected_classes=split_cfg['sel_cls_idx']
    for i in range(num_cls): #all_classes
        full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_imbclass_train'][i], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_val'][i], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_lake'][i], replace=False))
        else:
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_class_train'][i], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_val'][i], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_lake'][i], replace=False))
    
        train_idx += class_train_idx
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
    train_set = custom_subset(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = custom_subset(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = custom_subset(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    if(isnumpy):
        X = []
        for i in range(len(fullset)):
            X.append(fullset[i][0].tolist())
        X = np.array(X)
        X = X.transpose(0,2,3,1).astype(np.uint8)
        y = torch.from_numpy(np.array(fullset.targets))
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_unlabeled = X[lake_idx]
        y_unlabeled = y[lake_idx]
        return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set, selected_classes
    else:
        return train_set, val_set, lake_set, selected_classes

def create_longtail(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal):
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    selected_classes=split_cfg['sel_cls_idx']
    for i in range(num_cls): #all_classes
        full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_imbclass_train'][i], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_val'][i], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_lake'][i], replace=False))
        else:
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_class_train'][i], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_val'][i], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_lake'][i], replace=False))
    
        train_idx += class_train_idx
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
    train_set = custom_subset(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = custom_subset(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = custom_subset(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    if(isnumpy):
        if(dset_name=="mnist"):
            X  = fullset.data.numpy()
            y = torch.from_numpy(np.array(fullset.targets.float()))
        elif(dset_name=="svhn"):
            X = fullset.data
            y = torch.from_numpy(np.array(fullset.labels))
        else:            
            X = fullset.data
            y = torch.from_numpy(np.array(fullset.targets))
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_unlabeled = X[lake_idx]
        y_unlabeled = y[lake_idx]
        return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set, selected_classes
    else:
        return train_set, val_set, lake_set, selected_classes

def load_dataset_custom(datadir, dset_name, feature, split_cfg, isnumpy=False, augVal=False, dataAug=True):
    if(not(os.path.exists(datadir))):
        os.mkdir(datadir)

    if(dset_name=="cifar10"):
        num_cls=10
        cifar_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        if(dataAug):
            cifar_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            cifar_transform = cifar_test_transform
        
        fullset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=cifar_transform)
        test_set = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True, transform=cifar_test_transform)
        if(feature=="classimb"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set, imb_cls_idx = create_class_imb(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
            else:    
                train_set, val_set, lake_set, imb_cls_idx = create_class_imb(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        if(feature=="longtail"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set, imb_cls_idx = create_longtail(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
            else:    
                train_set, val_set, lake_set, imb_cls_idx = create_longtail(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        if(feature=="ood"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(dset_name, fullset, test_set, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']
            else:
                train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(dset_name, fullset, test_set, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
                return train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']
        if(feature=="vanilla"): 
            X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set = getVanillaData(dset_name, fullset, split_cfg)
            print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            if(isnumpy):
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls

        if(feature=="duplicate"):
            X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, lake_set = getDuplicateData(dset_name, fullset, split_cfg)
            print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            if(isnumpy):
                return X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls

    if(dset_name=="mnist"):
        num_cls=10
        mnist_test_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if(dataAug):
            mnist_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            mnist_transform = mnist_test_transform
        fullset = torchvision.datasets.MNIST(root=datadir, train=True, download=True, transform=mnist_transform)
        test_set = torchvision.datasets.MNIST(root=datadir, train=False, download=True, transform=mnist_test_transform)
        # fullset.data = torch.repeat_interleave(fullset.data.unsqueeze(1), 3, 1).float()
        if(feature=="classimb"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set, imb_cls_idx = create_class_imb(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal)
                print("MNIST Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
            else:    
                train_set, val_set, lake_set, imb_cls_idx = create_class_imb(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal)
                print("MNIST Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        if(feature=="ood"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(dset_name, fullset, test_set, split_cfg, num_cls, isnumpy, augVal)
                print("MNIST Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']
            else:
                train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(dset_name, fullset, test_set, split_cfg, num_cls, isnumpy, augVal)
                print("MNIST Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
                return train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']
        if(feature=="vanilla"): 
            X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set = getVanillaData(dset_name, fullset, split_cfg)
            print("MNIST Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            if(isnumpy):
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls

        if(feature=="duplicate"):
            X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, lake_set = getDuplicateData(dset_name, fullset, split_cfg)
            print("MNIST Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            print(y_tr)
            if(isnumpy):
                return X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls

    if(dset_name=="svhn"):
        num_cls=10
        SVHN_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        if(dataAug):
            SVHN_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        else:
            SVHN_transform = SVHN_test_transform
        
        fullset = torchvision.datasets.SVHN(root=datadir, split="train", download=True, transform=SVHN_transform)
        test_set = torchvision.datasets.SVHN(root=datadir, split="test", download=True, transform=SVHN_test_transform)
        if(feature=="classimb"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set, imb_cls_idx = create_class_imb(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal)
                print("SVHN Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
            else:    
                train_set, val_set, lake_set, imb_cls_idx = create_class_imb(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal)
                print("SVHN Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        if(feature=="ood"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(dset_name, fullset, test_set, split_cfg, num_cls, isnumpy, augVal)
                print("SVHN Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']
            else:
                train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(dset_name, fullset, test_set, split_cfg, num_cls, isnumpy, augVal)
                print("SVHN Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
                return train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']
        if(feature=="vanilla"): 
            X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set = getVanillaData(dset_name, fullset, split_cfg)
            print("SVHN Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            if(isnumpy):
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls

        if(feature=="duplicate"):
            X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, lake_set = getDuplicateData(dset_name, fullset, split_cfg)
            print("SVHN Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            if(isnumpy):
                return X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls

    if(dset_name=="cifar100"):
        num_cls=100
        cifar100_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        fullset = torchvision.datasets.CIFAR100(root=datadir, train=True, download=True, transform=cifar100_transform)
        test_set = torchvision.datasets.CIFAR100(root=datadir, train=False, download=True, transform=cifar100_transform)
        if(feature=="classimb"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set, imb_cls_idx = create_class_imb(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
            else:    
                train_set, val_set, lake_set, imb_cls_idx = create_class_imb(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        if(feature=="ood"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(dset_name, fullset, test_set, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, ood_cls_idx, num_cls
            else:
                train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(dset_name, fullset, test_set, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
                return train_set, val_set, test_set, lake_set, ood_cls_idx, num_cls
        
        if(feature=="vanilla"):
            X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set = getVanillaData(dset_name, fullset, split_cfg)
            print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            if(isnumpy):
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls

        if(feature=="duplicate"):
            X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, lake_set = getDuplicateData(dset_name, fullset, split_cfg)
            print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            if(isnumpy):
                return X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls
            
    
    if(dset_name=="breast_density"):
        num_cls=4
        data_dir = datadir
#         data_dir = "/home/snk170001/research/data/custom/bc_train_test"
        input_size=224
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        fullset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
        test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
        if(feature=="classimb"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set, imb_cls_idx = create_class_imb_bio(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal)
                print("Breast-density Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
            else:    
                train_set, val_set, lake_set, imb_cls_idx = create_class_imb_bio(dset_name, fullset, split_cfg, num_cls, isnumpy, augVal)
                print("Breast-density Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls