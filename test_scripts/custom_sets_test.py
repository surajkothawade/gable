from gable.utils.custom_dataset import *
from gable.utils.FairFace import FairFace
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt

datadir = "C:/Users/nbeck/Documents/pytorch_data"
dset_name = "fairface"
feature = "attrimb"

# Set parameters like the class imbalance function. Here, we must provide an attribute domain size.
# It specifies the number of values that the imbalance attribute may take. In the case of the race 
# attribute in FairFace, race can take 4 different values, so attr_domain_size is set to 4.
isnumpy = True
augVal = False

# The following shows what the split config should look like. Here, we specify the attribute to imbalance 
# to be the race attribute. For FairFace, it can be one of {race, gender, age}. Note that choosing the 
# imbalance attribute to be the same as the target attribute would be equivalent to doing a class imbalance.
split_cfg = {
    "target_attr": "age",
    "attr": "race",                # The attribute to imbalance. Must be the name of the attribute of the Dataset class.
    "attr_dom_size": 7,            # The number of classes of that attribute
    "attr_imb_cls": [0,1],           # The specific attribute classes to imbalance
    "per_attr_imb_train": 1,      # The number of training data points to keep for each affected value of the imbalance attribute domain
    "per_attr_imb_val": 1,        # The number of validation data points to keep for each affected value of the imbalance attribute domain
    "per_attr_imb_test": 1,        # The number of test data points to keep for each affected value of the imbalance attribute domain
    "per_attr_imb_lake": 1,       # The number of lake data points to keep for each affected value of the imbalance attribute domain
    "per_attr_train": 2,          # The number of training data points to keep for each unaffected value of the imbalance attribute domain
    "per_attr_val": 2,            # The number of validation data points to keep for each unaffected value of the imbalance attribute domain
    "per_attr_test": 2,            # The number of test data points to keep for each unaffected value of the imbalance attribute domain
    "per_attr_lake": 2            # The number of lake data points to keep for each unaffected value of the imbalance attribute domain
    }

# We perform the attribute imbalance
X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, selected_classes, num_cls = load_dataset_custom(datadir, dset_name, feature, split_cfg, isnumpy, augVal)

# For the train_set, get two custom subsets
idx_subset_1 = [0,1,2,3]
idx_subset_2 = [4,5,6,7]
lab_subset_1 = train_set.targets[idx_subset_1]
lab_subset_2 = train_set.targets[idx_subset_2]

train_subset_1 = custom_subset(train_set, idx_subset_1, lab_subset_1, train_set.age, train_set.race, train_set.gender)
train_subset_2 = custom_subset(train_set, idx_subset_2, lab_subset_2, train_set.age, train_set.race, train_set.gender)

print("SUBSET 1 BEFORE CONSTRUCTION")
print("Label:", lab_subset_1)
print("Age:", train_set.age[idx_subset_1])
print("Race", train_set.race[idx_subset_1])
print("Gender:", train_set.gender[idx_subset_1])
print("========")
print("SUBSET 2 BEFORE CONSTRUCTION")
print("Label:", lab_subset_2)
print("Age:", train_set.age[idx_subset_2])
print("Race", train_set.race[idx_subset_2])
print("Gender:", train_set.gender[idx_subset_2])
print("========")

print("SUBSET 1 AFTER CONSTRUCTION")
print("Label:", train_subset_1.targets)
print("Age:", train_subset_1.age)
print("Race", train_subset_1.race)
print("Gender:", train_subset_1.gender)
print("========")
print("SUBSET 2 AFTER CONSTRUCTION")
print("Label:", train_subset_2.targets)
print("Age:", train_subset_2.age)
print("Race", train_subset_2.race)
print("Gender:", train_subset_2.gender)
print("========")

train_concatenated_subsets = custom_concat(train_subset_1, train_subset_2)

print("Concatenated Subsets")
print("Label:", train_concatenated_subsets.targets)
print("Age:", train_concatenated_subsets.age)
print("Race", train_concatenated_subsets.race)
print("Gender:", train_concatenated_subsets.gender)