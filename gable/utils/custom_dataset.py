import numpy as np
import os
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image as Image
import copy
import math

from .UTKFace import UTKFace
from .FairFace import FairFace

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



    def __init__(self, dataset, indices, labels, age_attributes=None, race_attributes=None, gender_attributes=None):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels.type(torch.long)
        if age_attributes is not None: self.age = age_attributes[indices]
        if race_attributes is not None: self.race = race_attributes[indices]
        if gender_attributes is not None: self.gender = gender_attributes[indices]
        self.indices = np.array(indices)

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

    def get_attr_val(self, attribute, indices):
        # Obtain the target attribute to imbalance
        imbalance_attribute = getattr(self, attribute)

        # Return only the imbalance attribute values that correspond to the indices.
        return imbalance_attribute[indices]

class custom_concat(Dataset):
    r"""
    Concat of a dataset at specified indices.
    """
    def __init__(self, dataset1, dataset2):
        self.dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
        self.targets = torch.Tensor(list(dataset1.targets) + list(dataset2.targets)).type(torch.long)
    
        if getattr(dataset1, "age", None) is not None and getattr(dataset2, "age", None) is not None: self.age = np.concatenate((dataset1.age, dataset2.age), axis=0)
        if getattr(dataset1, "race", None) is not None and getattr(dataset2, "race", None) is not None: self.race = np.concatenate((dataset1.race, dataset2.race), axis=0)
        if getattr(dataset1, "gender", None) is not None and getattr(dataset2, "gender", None) is not None: self.gender = np.concatenate((dataset1.gender, dataset2.gender), axis=0)
    
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)
    
    def get_attr_val(self, attribute, indices):
        # Obtain the target attribute to imbalance
        imbalance_attribute = getattr(self, attribute)

        # Return only the imbalance attribute values that correspond to the indices.
        return imbalance_attribute[indices]

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
            return x, y, index

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return x, index

    def __len__(self):
        return len(self.X)
   
class DataHandler_FairFace(Dataset):
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
            x = Image.fromarray(x)
            x = self.transform(x)
            return x, y, index

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


def curate_test_set(fullset, example_factor=1, ignore_idx=None):
        
    # We want a balance between age, race, gender in our test set.
    # Meaning for a specific age, race, and gender tuple, we should 
    # have a uniform distribution.
    test_set_idx = []
        
    full_idx = [x for x in range(len(fullset.age))]
    
    np.random.shuffle(full_idx)
    
    for i in range(fullset.n_age):
        ages_to_search = np.where(fullset.age == i)[0]
        ages_idx = np.array(full_idx)[ages_to_search]
        for j in range(fullset.n_gender):
            gender_project = fullset.gender[ages_idx]
            age_genders_to_search = np.where(gender_project == j)[0]
            age_genders_idx = np.array(ages_idx)[age_genders_to_search]
            for k in range(fullset.n_race):
                age_gender_project = fullset.race[age_genders_idx]
                age_gender_races_to_search = np.where(age_gender_project == k)[0]
                age_gender_races_idx = np.array(age_genders_idx)[age_gender_races_to_search]
                l = 0
                while l < example_factor:
                    if l >= len(age_gender_races_idx):
                        break
                    add_index = age_gender_races_idx[l]
                    
                    if ignore_idx is not None:
                        if add_index in ignore_idx:
                            continue
                    
                    test_set_idx.append(add_index)
                    l += 1
        
    return test_set_idx

def create_attr_imb(fullset, split_cfg, attr_domain_size, isnumpy, augVal):
    
    # Set random seed to ensure reproducibility
    np.random.seed(43)
    
    # Selection idx for train, val, lake sets
    train_idx = []
    val_idx = []
    lake_idx = []
    test_idx = []
    
    # Get specific values of the imbalance attribute to apply the imbalance
    selected_attribute_classes = np.array(split_cfg['attr_imb_cls'])
    
    # Obtain the target attribute to imbalance
    imbalance_attribute = getattr(fullset, split_cfg['attr'])
    
    # Before the other sets are constructed, curate a test set.
    test_idx = curate_test_set(fullset, example_factor=split_cfg['test_set_size_mult'])
    
    # Curate an initial train set that is also balanced in the same fashion (but randomized).
    nclasses = split_cfg['attr_dom_size']
    temp_train_size = nclasses * split_cfg['per_attr_train']
    expected_train_size = len(split_cfg['attr_imb_cls']) * split_cfg['per_attr_imb_train'] + (nclasses - len(split_cfg["attr_imb_cls"])) * split_cfg['per_attr_train']
    one_factor_test_size = fullset.n_age * fullset.n_gender * fullset.n_race
    train_set_factor = math.ceil(temp_train_size / one_factor_test_size)
    train_idx = curate_test_set(fullset, example_factor=train_set_factor, ignore_idx=test_idx)
    
    if len(train_idx) > expected_train_size:
        for imbalanced_attribute_val in range(nclasses):
            
            # Achieve the requested number of imbalanced samples.
            proj_imbalance_attribute = imbalance_attribute[train_idx]
            
            if imbalanced_attribute_val in split_cfg['attr_imb_cls']:
                
                # Go to next iteration of loop if this class already has the equal per_attr_imb_train.
                imb_attr_val_idx = np.where(proj_imbalance_attribute == imbalanced_attribute_val)[0].tolist()
                if len(imb_attr_val_idx) == split_cfg['per_attr_imb_train']:
                    continue
                
                # Otherwise, pick random indices to delete
                num_remove = len(imb_attr_val_idx) - split_cfg['per_attr_imb_train']
                np.random.shuffle(imb_attr_val_idx)
                indices_to_trim = np.array(imb_attr_val_idx)[list(range(num_remove))]
                train_idx = np.setdiff1d(train_idx, np.array(train_idx)[indices_to_trim])
            else:
                # Go to next iteration of loop if this class already has the equal per_attr_imb_train.
                imb_attr_val_idx = np.where(proj_imbalance_attribute == imbalanced_attribute_val)[0].tolist()
                if len(imb_attr_val_idx) == split_cfg['per_attr_train']:
                    continue
                
                # Otherwise, pick random indices to delete
                num_remove = len(imb_attr_val_idx) - split_cfg['per_attr_train']
                np.random.shuffle(imb_attr_val_idx)
                indices_to_trim = np.array(imb_attr_val_idx)[list(range(num_remove))]
                train_idx = np.setdiff1d(train_idx, np.array(train_idx)[indices_to_trim])
    
    # Loop over all classes of the attribute domain
    for i in range(attr_domain_size):
        full_idx_attr_class = torch.where(torch.Tensor(imbalance_attribute) == i)[0].cpu().numpy()
        
        # Make sure to exclude those already chosen by the test set.
        full_idx_attr_class = list(np.setdiff1d(full_idx_attr_class, np.array(test_idx)))
        
        # Do not bother with attribute classes that have no elements to choose.
        if len(full_idx_attr_class) == 0:
            continue
        
        # If the attribute was chosen to be imbalanced, select a random subset of the imbalanced size for train, val, lake sets.
        # Otherwise, select random subsets of the default size for train, test, val sets.
        if i in selected_attribute_classes:
            attr_class_val_idx = list(np.random.choice(np.array(full_idx_attr_class), size=split_cfg['per_attr_imb_val'], replace=False))
            remain_idx = list(set(full_idx_attr_class) - set(attr_class_val_idx))
            attr_class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_attr_imb_lake'], replace=False))
            remain_idx = list(set(remain_idx) - set(attr_class_lake_idx))
        else:
            attr_class_val_idx = list(np.random.choice(np.array(full_idx_attr_class), size=split_cfg['per_attr_val'], replace=False))
            remain_idx = list(set(full_idx_attr_class) - set(attr_class_val_idx))
            attr_class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_attr_lake'], replace=False))
            remain_idx = list(set(remain_idx) - set(attr_class_lake_idx))
            
        # Add selected idx to each set. If augVal, then augment training set 
        # with validation samples from the imbalanced attribute classes     
        if augVal and i in selected_attribute_classes:
            train_idx += attr_class_val_idx
        val_idx += attr_class_val_idx
        lake_idx += attr_class_lake_idx

    # Create custom subsets for each set
    train_set = custom_subset(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx], age_attributes=fullset.age, race_attributes=fullset.race, gender_attributes=fullset.gender)
    val_set = custom_subset(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx], age_attributes=fullset.age, race_attributes=fullset.race, gender_attributes=fullset.gender)
    lake_set = custom_subset(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx], age_attributes=fullset.age, race_attributes=fullset.race, gender_attributes=fullset.gender)
    test_set = custom_subset(fullset, test_idx, torch.Tensor(fullset.targets)[test_idx], age_attributes=fullset.age, race_attributes=fullset.race, gender_attributes=fullset.gender)
    
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
        RESIZE_DIM=192
        
        # Pull transform from above
        utkface_transform = transforms.Compose([
            transforms.Resize((RESIZE_DIM, RESIZE_DIM)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        if "age_bins" in split_cfg:
            age_bins = split_cfg["age_bins"]
        else:
            age_bins = None
        
        # Get UTKFace
        fullset = UTKFace(root=datadir, target_type=split_cfg["target_attr"], download=True, transform=utkface_transform, age_bins=age_bins)
        num_cls = fullset.nclasses
        
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
      
    # NEEDS A COUPLE CHANGES BEFORE IT CAN BE IN USE.
    if(dset_name=="fairface"):
        # Pull transform from above
        utkface_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Get UTKFace
        fullset = FairFace(root=datadir, target_type=split_cfg["target_attr"], download=True, transform=utkface_transform, load_cap=50000)
        num_cls = fullset.nclasses
        
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