from gable.utils.custom_dataset import load_dataset_custom
from gable.utils.FairFace import FairFace
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

datadir = "C:/Users/nbeck/Documents/pytorch_data"
dset_name = "fairface"
feature = "attrimb"

fair_face_dataset = FairFace(datadir, download=True, load_cap = 20000)

for i in range(10):
    
    image, target = fair_face_dataset[i]
    
    print(i)
    print(image)
    print(target)
    
    
    plt.imshow(image)
    plt.show()

"""
# Set parameters like the class imbalance function. Here, we must provide an attribute domain size.
# It specifies the number of values that the imbalance attribute may take. In the case of the race 
# attribute in UTKFace, race can take 4 different values, so attr_domain_size is set to 4.
isnumpy = True
augVal = False

# The following shows what the split config should look like. Here, we specify the attribute to imbalance 
# to be the race attribute. For UTKFace, it can be one of {race, gender, age}. Note that choosing the 
# imbalance attribute to be the same as the target attribute would be equivalent to doing a class imbalance.
split_cfg = {
    "attr": "gender",                # The attribute to imbalance. Must be the name of the attribute of the Dataset class.
    "attr_dom_size": 2,            # The number of classes of that attribute
    "attr_imb_cls": [1],           # The specific attribute classes to imbalance
    "per_attr_imb_train": 10,      # The number of training data points to keep for each affected value of the imbalance attribute domain
    "per_attr_imb_val": 11,        # The number of validation data points to keep for each affected value of the imbalance attribute domain
    "per_attr_imb_test": 9,        # The number of test data points to keep for each affected value of the imbalance attribute domain
    "per_attr_imb_lake": 12,       # The number of lake data points to keep for each affected value of the imbalance attribute domain
    "per_attr_train": 21,          # The number of training data points to keep for each unaffected value of the imbalance attribute domain
    "per_attr_val": 22,            # The number of validation data points to keep for each unaffected value of the imbalance attribute domain
    "per_attr_test": 9,            # The number of test data points to keep for each unaffected value of the imbalance attribute domain
    "per_attr_lake": 23            # The number of lake data points to keep for each unaffected value of the imbalance attribute domain
    }

# We perform the attribute imbalance
X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, selected_classes, num_cls = load_dataset_custom(datadir, dset_name, feature, split_cfg, isnumpy, augVal)
print("TRAINING LABEL/ATTRIBUTE")
print(y_tr)

print()
print("VALIDATION LABEL/ATTRIBUTE")
print(y_val)

print()
print("LAKE LABEL/ATTRIBUTE")
print(y_unlabeled)

test_dl = DataLoader(test_set, batch_size=1)

for i, (x,y) in enumerate(test_dl):
    print(x.shape)
"""