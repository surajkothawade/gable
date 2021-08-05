# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 21:22:34 2021

@author: nbeck
"""

import numpy as np
import torch
import torchvision

from torchvision import transforms
from .custom_utils import custom_subset

def get_ood_targets(targets, num_idc_cls):
    ood_targets = []
    targets_list = list(targets)
    for i in range(len(targets_list)):
        if targets_list[i] < num_idc_cls:
            ood_targets.append(targets_list[i])
        else:
            ood_targets.append(num_idc_cls)
    print("num ood samples: ", ood_targets.count(num_idc_cls))
    return torch.Tensor(ood_targets)

def cifar10_dataset_combo_perturbation(root, split_cfg, isnumpy, augVal, dataAug):
    
    cifar_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    if dataAug:
        cifar_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        cifar_transform = cifar_test_transform
    
    # Get CIFAR10
    dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=cifar_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=cifar_test_transform)
    num_cls = 10
    
    # Specify some classes from the dataset as being rare
    rare_class_list = split_cfg['rare_classes']
    
    # Specify some classes from the dataset as constituting OOD classes. These are the end classes of the dataset.
    num_idc_classes = split_cfg['num_idc_classes']
    ood_class_list = list(range(split_cfg['num_idc_classes'], num_cls))
    
    # Ensure that these classes do not intersect
    if len(set(rare_class_list).intersection(set(ood_class_list))) > 0:
        raise ValueError("Rare/OOD class mismatch")
    
    # Get the labels for this dataset as a torch tensor
    dataset_labels = torch.Tensor(dataset.targets)
      
    train_idx = []
    val_idx = []
    lake_idx = []
    
    # Loop through all classes, obtaining relevant points (specifically, their indices) satisfying the split config.
    for i in range(num_cls):
        
        # Get only those labels specific to the examined class.
        class_idx = list(torch.where(dataset_labels == i)[0].cpu().numpy())
        
        # Determine how many points to select for each set from this class
        if i in rare_class_list:
            num_from_class_to_select_for_train = split_cfg['pc_rare_train']
            num_from_class_to_select_for_val = split_cfg['pc_rare_val']
            num_from_class_to_select_for_lake = split_cfg['pc_rare_lake']
        elif i in ood_class_list:
            num_from_class_to_select_for_train = split_cfg['pc_ood_train']
            num_from_class_to_select_for_val = split_cfg['pc_ood_val']
            num_from_class_to_select_for_lake = split_cfg['pc_ood_lake']
        else:
            num_from_class_to_select_for_train = split_cfg['pc_normal_train']
            num_from_class_to_select_for_val = split_cfg['pc_normal_val']
            num_from_class_to_select_for_lake = split_cfg['pc_normal_lake']
            
        # Select random indices from this class
        class_train_idx = list(np.random.choice(class_idx, size=num_from_class_to_select_for_train, replace=False))
        class_idx = list(set(class_idx) - set(class_train_idx))
        class_val_idx = list(np.random.choice(class_idx, size=num_from_class_to_select_for_val, replace=False))
        class_idx = list(set(class_idx) - set(class_val_idx))
        class_lake_idx = list(np.random.choice(class_idx, size=num_from_class_to_select_for_lake, replace=False))
        
        # We can do the duplication step here. If i represents a normal class, we can duplicate some of the indices 
        # in the lake. When we go to create a subset, this actually allows us to artificially duplicate points.
        
        #if i not in rare_class_list and i not in ood_class_list:
            
        num_lake_duplicate_repetitions = split_cfg['num_rep']
        lake_duplicated_subset_size = split_cfg['per_normal_class_lake_duplicated_subset_size']
        class_lake_idx = class_lake_idx[lake_duplicated_subset_size:] + class_lake_idx[:lake_duplicated_subset_size] * num_lake_duplicate_repetitions
            
        # Some instances require augmenting the train set with examples from the validation set. This is done here.
        if augVal:
            if i in rare_class_list or i not in ood_class_list:
                train_idx += class_val_idx
            
        # Add this bout of selection to the running lists for each set
        train_idx += class_train_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
        
    # Now that we have the index lists for each set, we now must fix each point's label due to the presence of OOD.
    train_set_labels = get_ood_targets(dataset_labels[train_idx], num_idc_classes)
    val_set_labels = get_ood_targets(dataset_labels[val_idx], num_idc_classes)
    lake_set_labels = get_ood_targets(dataset_labels[lake_idx], num_idc_classes)
    
    # Lastly, we make custom subsets for each set.
    train_set = custom_subset(dataset, train_idx, train_set_labels)
    val_set = custom_subset(dataset, val_idx, val_set_labels)
    lake_set = custom_subset(dataset, lake_idx, lake_set_labels)
    
    # We also need to fix the test set's labels due to the presence of OOD.
    keep_test_idx = list(torch.where(torch.Tensor(test_dataset.targets) < num_idc_classes)[0].cpu().numpy())
    test_dataset_labels = torch.Tensor(test_dataset.targets)[keep_test_idx]
    test_set = custom_subset(test_dataset, keep_test_idx, test_dataset_labels)
    
    if isnumpy:
        X_tr = dataset.data[train_idx]
        y_tr = train_set_labels
        X_val = dataset.data[val_idx]
        y_val = val_set_labels
        X_lake = dataset.data[lake_idx]
        y_lake = lake_set_labels
        
        return X_tr, y_tr, X_val, y_val, X_lake, y_lake, train_set, val_set, test_set, lake_set
    else:
        return train_set, val_set, test_set, lake_set