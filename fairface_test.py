from gable.utils.custom_dataset import load_dataset_custom
from gable.utils.FairFace import FairFace
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

datadir = "C:/Users/nbeck/Documents/pytorch_data"
dset_name = "fairface"
feature = "attrimb"

"""
load_cap = 500
fair_face_dataset = FairFace(datadir, download=True, load_cap = load_cap)

age_freq = [0 for x in range(9)]
race_freq = [0 for x in range(7)]
gend_freq = [0 for x in range(2)]

for i in range(load_cap):
    
    image, target = fair_face_dataset[i]
    
    age = fair_face_dataset.age[i]
    race = fair_face_dataset.race[i]
    gender = fair_face_dataset.gender[i]
    print(F"{i+1} (a,r,g): {age}, {gender}, {race}")
    age_freq[fair_face_dataset.age[i]] += 1
    race_freq[fair_face_dataset.race[i]] += 1
    gend_freq[fair_face_dataset.gender[i]] += 1


plt.imshow(fair_face_dataset.data[0])

print(age_freq)
print(race_freq)
print(gend_freq)
"""


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
    "per_attr_imb_lake": 1,       # The number of lake data points to keep for each affected value of the imbalance attribute domain
    "per_attr_train": 2,          # The number of training data points to keep for each unaffected value of the imbalance attribute domain
    "per_attr_val": 2,            # The number of validation data points to keep for each unaffected value of the imbalance attribute domain
    "per_attr_lake": 2,            # The number of lake data points to keep for each unaffected value of the imbalance attribute domain
    "test_set_size_mult": 3
    }

# We perform the attribute imbalance
X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, selected_classes, num_cls = load_dataset_custom(datadir, dset_name, feature, split_cfg, isnumpy, augVal)

age_freq = [0 for x in range(9)]
race_freq = [0 for x in range(7)]
gend_freq = [0 for x in range(2)]
test_this_set = test_set
test_this_data = X_val
test_this_label = y_val
for i in range(len(test_this_set)):
    
    #plt.imshow(test_this_data[i])
    #label = test_this_label[i]
    #print(F"{i+1} label: {label}")
    plt.show()
    age = test_this_set.age[i]
    race = test_this_set.race[i]
    gender = test_this_set.gender[i]
    print(F"{i+1} (a,g,r): {age}, {gender}, {race}")
    age_freq[test_this_set.age[i]] += 1
    race_freq[test_this_set.race[i]] += 1
    gend_freq[test_this_set.gender[i]] += 1
    
print(age_freq)
print(race_freq)
print(gend_freq)

