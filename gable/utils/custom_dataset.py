import numpy as np
import os
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image as Image

from .UTKFace import UTKFace

np.random.seed(42)
torch.manual_seed(42)

class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels.type(torch.long)
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

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
        RESIZE_DIM = 192
        if(use_test_transform):
            transform = transforms.Compose([transforms.Resize((RESIZE_DIM, RESIZE_DIM)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std
        else:
            transform = transforms.Compose([transforms.Resize((RESIZE_DIM, RESIZE_DIM)), transforms.RandomCrop(200, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std
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
            return (x, y), index

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return x, index

    def __len__(self):
        return len(self.X)
    

def create_ood_data(fullset, testset, split_cfg, num_cls, isnumpy, augVal):
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    test_idx = []
    selected_classes = np.random.choice(np.arange(num_cls), size=split_cfg['num_cls_idc'], replace=False) #number of in distribution classes
    # selected_classes = list(range(split_cfg['num_cls_idc']))
    for i in range(num_cls): #all_classes
        full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
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
    
    train_set = custom_subset(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = custom_subset(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = custom_subset(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    test_set = custom_subset(testset, test_idx, torch.Tensor(testset.targets)[test_idx])
    if(isnumpy):        
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

def create_attr_imb(fullset, split_cfg, attr_domain_size, isnumpy, augVal):
    
    # Set random seed to ensure reproducibility
    np.random.seed(42)
    
    # Selection idx for train, val, lake sets
    train_idx = []
    val_idx = []
    lake_idx = []
    test_idx = []
    
    # Get specific values of the imbalance attribute to apply the imbalance
    selected_attribute_classes = np.array(split_cfg['attr_imb_cls'])
    
    # Obtain the target attribute to imbalance
    imbalance_attribute = getattr(fullset, split_cfg['attr'])
    
    # Loop over all classes of the attribute domain
    for i in range(attr_domain_size):
        full_idx_attr_class = list(torch.where(torch.Tensor(imbalance_attribute) == i)[0].cpu().numpy())
        
        # Do not bother with attribute classes that have no elements to choose.
        if len(full_idx_attr_class) == 0:
            continue
        
        # If the attribute was chosen to be imbalanced, select a random subset of the imbalanced size for train, val, lake sets.
        # Otherwise, select random subsets of the default size for train, test, val sets.
        if i in selected_attribute_classes:
            attr_class_train_idx = list(np.random.choice(np.array(full_idx_attr_class), size=split_cfg['per_attr_imb_train'], replace=False))
            remain_idx = list(set(full_idx_attr_class) - set(attr_class_train_idx))
            attr_class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_attr_imb_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(attr_class_val_idx))
            attr_class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_attr_imb_lake'], replace=False))
            remain_idx = list(set(remain_idx) - set(attr_class_lake_idx))
            attr_class_test_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_attr_imb_test'], replace=False))
            remain_idx = list(set(remain_idx) - set(attr_class_test_idx))
        else:
            attr_class_train_idx = list(np.random.choice(np.array(full_idx_attr_class), size=split_cfg['per_attr_train'], replace=False))
            remain_idx = list(set(full_idx_attr_class) - set(attr_class_train_idx))
            attr_class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_attr_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(attr_class_val_idx))
            attr_class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_attr_lake'], replace=False))
            remain_idx = list(set(remain_idx) - set(attr_class_lake_idx))
            attr_class_test_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_attr_test'], replace=False))
            remain_idx = list(set(remain_idx) - set(attr_class_test_idx))

        # Add selected idx to each set. If augVal, then augment training set 
        # with validation samples from the imbalanced attribute classes     
        train_idx += attr_class_train_idx
        if augVal and i in selected_attribute_classes:
            train_idx += attr_class_val_idx
        val_idx += attr_class_val_idx
        lake_idx += attr_class_lake_idx
        test_idx += attr_class_test_idx

    # Create custom subsets for each set
    train_set = custom_subset(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = custom_subset(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = custom_subset(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    test_set = custom_subset(fullset, test_idx, torch.Tensor(fullset.targets)[test_idx])    

    # If specified, create and return additional numpy arrays. Otherwise, just return custom subsets and selected attribute classes
    if isnumpy:
        X = fullset.data
        y = torch.from_numpy(np.array(fullset.targets))
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_unlabeled = X[lake_idx]
        y_unlabeled = y[lake_idx]
        
        return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, selected_attribute_classes
    else:
        return train_set, val_set, test_set, lake_set, selected_attribute_classes

def create_class_imb(fullset, split_cfg, num_cls, isnumpy, augVal):
    np.random.seed(42)
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
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx

    train_set = custom_subset(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = custom_subset(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = custom_subset(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    if(isnumpy):        
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

def getDuplicateData(fullset, split_cfg):
    num_rep=split_cfg['num_rep']
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
    train_set = DataHandler_CIFAR10(X_tr, y_tr, False)
    lake_set = DataHandler_CIFAR10(X_unlabeled_rep, y_unlabeled_rep, False)
    val_set = DataHandler_CIFAR10(X_val, y_val, False)
    return X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, lake_set

def getVanillaData(fullset, split_cfg):
    X = fullset.data
    y = torch.from_numpy(np.array(fullset.targets))
    X_tr = X[:split_cfg['train_size']]
    y_tr = y[:split_cfg['train_size']]
    X_unlabeled = X[split_cfg['train_size']:len(X)-split_cfg['val_size']]
    y_unlabeled = y[split_cfg['train_size']:len(X)-split_cfg['val_size']]
    X_val = X[len(X)-split_cfg['val_size']:]
    y_val = y[len(X)-split_cfg['val_size']:]
    train_set = DataHandler_CIFAR10(X_tr, y_tr, False)
    lake_set = DataHandler_CIFAR10(X_unlabeled[:split_cfg['lake_size']], y_unlabeled[:split_cfg['lake_size']], False)
    val_set = DataHandler_CIFAR10(X_val, y_val, False)
    return X_tr, y_tr, X_val, y_val, X_unlabeled[:split_cfg['lake_size']], y_unlabeled[:split_cfg['lake_size']], train_set, val_set, lake_set

#TODO: Add attimb, duplicates, weak aug, out-of-dist settings
def load_dataset_custom(datadir, dset_name, feature, split_cfg, isnumpy=False, augVal=False):
    if(not(os.path.exists(datadir))):
        os.mkdir(datadir)

    if(dset_name=="cifar10"):
        num_cls=10
        cifar_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        cifar_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
        fullset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=cifar_transform)
        test_set = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True, transform=cifar_test_transform)
        if(feature=="classimb"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set, imb_cls_idx = create_class_imb(fullset, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
            else:    
                train_set, val_set, lake_set, imb_cls_idx = create_class_imb(fullset, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        if(feature=="ood"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(fullset, test_set, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, ood_cls_idx, num_cls
            else:
                train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(fullset, test_set, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
                return train_set, val_set, test_set, lake_set, ood_cls_idx, num_cls
        if(feature=="vanilla"): 
            X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set = getVanillaData(fullset, split_cfg)
            print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            if(isnumpy):
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls

        if(feature=="duplicate"):
            X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, lake_set = getDuplicateData(fullset, split_cfg)
            print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            if(isnumpy):
                return X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls

    if(dset_name=="mnist"):
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
            print("MNIST Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))

            return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls

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
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set, imb_cls_idx = create_class_imb(fullset, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
            else:    
                train_set, val_set, lake_set, imb_cls_idx = create_class_imb(fullset, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        if(feature=="ood"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(fullset, test_set, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, ood_cls_idx, num_cls
            else:
                train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(fullset, test_set, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
                return train_set, val_set, test_set, lake_set, ood_cls_idx, num_cls
        
        if(feature=="vanilla"):
            X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set = getVanillaData(fullset, split_cfg)
            print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            if(isnumpy):
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls

        if(feature=="duplicate"):
            X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, lake_set = getDuplicateData(fullset, split_cfg)
            print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            if(isnumpy):
                return X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls

    if(dset_name=="utkface"):
        # We are targeting the age class
        num_cls=117
        RESIZE_DIM=192
        
        # Pull transform from above
        utkface_transform = transforms.Compose([
            transforms.Resize((RESIZE_DIM, RESIZE_DIM)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Get UTKFace
        fullset = UTKFace(root=datadir, download=True, transform=utkface_transform)
        
        # Currently, only supporting attribute imbalance as that is the purpose of the UTKFace dataset here.
        if(feature=="attrimb"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, imb_attr_cls_idx = create_attr_imb(fullset, split_cfg, split_cfg['attr_dom_size'], isnumpy, augVal)
                print("UTKFace Custom dataset stats: Train size:", len(train_set), "Val size:", len(val_set), "Test size:", len(test_set), "Lake size:", len(lake_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, imb_attr_cls_idx, num_cls
            else:
                train_set, val_set, test_set, lake_set, imb_attr_cls_idx = create_attr_imb(fullset, split_cfg, split_cfg['attr_dom_size'], isnumpy, augVal)
                print("UTKFace Custom dataset stats: Train size:", len(train_set), "Val size:", len(val_set), "Test size:", len(test_set), "Lake size:", len(lake_set))
                return train_set, val_set, test_set, lake_set, imb_attr_cls_idx, num_cls