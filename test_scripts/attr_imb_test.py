from gable.utils.custom_dataset import create_attr_imb
from gable.utils.UTKFace import UTKFace

import torch
import numpy as np

# Load the dataset, providing a root directory, a target type which represents the attribute to treat as the label, 
# and whether or not to download the dataset. For UTKFace, target_type can be one of {"age", "race", "gender"}.
# Based on our discussion, we wanted to test age prediction with regards to a race imbalance.
dataset = UTKFace("C:/Users/nbeck/Documents/pytorch_data", target_type="age", download=True)

# Set parameters like the class imbalance function. Here, we must provide an attribute domain size.
# It specifies the number of values that the imbalance attribute may take. In the case of the race 
# attribute in UTKFace, race can take 4 different values, so attr_domain_size is set to 4.
attr_domain_size = 4
isnumpy = True
augVal = False

# The following shows what the split config should look like. Here, we specify the attribute to imbalance 
# to be the race attribute. For UTKFace, it can be one of {race, gender, age}. Note that choosing the 
# imbalance attribute to be the same as the target attribute would be equivalent to doing a class imbalance.
split_cfg = {
    "attr": "race",                # The attribute to imbalance. Must be the name of the attribute of the Dataset class.
    "num_attr_cls_imb": 1,         # The number of classes of the imbalance attribute domain to imbalance
    "per_attr_imb_train": 10,      # The number of training data points to keep for each affected value of the imbalance attribute domain
    "per_attr_imb_val": 11,        # The number of validation data points to keep for each affected value of the imbalance attribute domain
    "per_attr_imb_lake": 12,       # The number of lake data points to keep for each affected value of the imbalance attribute domain
    "per_attr_train": 21,          # The number of training data points to keep for each unaffected value of the imbalance attribute domain
    "per_attr_val": 22,            # The number of validation data points to keep for each unaffected value of the imbalance attribute domain
    "per_attr_lake": 23            # The number of lake data points to keep for each unaffected value of the imbalance attribute domain
    }

# We perform the attribute imbalance
X_tr, y_tr, a_tr, X_val, y_val, a_val, X_unlabeled, y_unlabeled, a_unlabeled, train_set, val_set, lake_set, selected_classes = create_attr_imb(dataset, split_cfg, attr_domain_size, isnumpy, augVal)

print("TRAINING LABEL/ATTRIBUTE")
print(y_tr)
print(a_tr)

print()
print("VALIDATION LABEL/ATTRIBUTE")
print(y_val)
print(a_val)

print()
print("LAKE LABEL/ATTRIBUTE")
print(y_unlabeled)
print(a_unlabeled)

# Show which values of the imbalance attribute domain were affected
print()
print("Imbalanced Attribute Classes:", selected_classes)

# Print the training split
print("TRAINING SPLITS")
for i in range(attr_domain_size):
    num_attr_trn = torch.where(a_tr == i)[0].cpu().numpy().shape[0]
    print("Class", i, "has", num_attr_trn, "points")
    
# Print the validation split
print()
print("VAL SPLITS")
for i in range(attr_domain_size):
    num_attr_trn = torch.where(a_val == i)[0].cpu().numpy().shape[0]
    print("Class", i, "has", num_attr_trn, "points")
    
# Print the lake split
print()
print("LAKE SPLITS")
for i in range(attr_domain_size):
    num_attr_trn = torch.where(a_unlabeled == i)[0].cpu().numpy().shape[0]
    print("Class", i, "has", num_attr_trn, "points")