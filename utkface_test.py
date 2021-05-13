from gable.utils.custom_dataset import load_dataset_custom
from gable.utils.UTKFace import UTKFace
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

datadir = "C:/Users/nbeck/Documents/pytorch_data"
dset_name = "utkface"
feature = "attrimb"

"""
age_bins = [3*(1+x) for x in range(40)]
utk_face_dataset = UTKFace(datadir, download=True, age_bins=age_bins)

age_freq = [0 for x in range(utk_face_dataset.n_age)]
race_freq = [0 for x in range(utk_face_dataset.n_race)]
gend_freq = [0 for x in range(utk_face_dataset.n_gender)]

for i in range(len(utk_face_dataset.test_age)):
    age = utk_face_dataset.test_age[i]
    race = utk_face_dataset.test_race[i]
    gender = utk_face_dataset.test_gender[i]
    #print(F"{i+1} (a,r,g): {age}, {gender}, {race}")
    age_freq[utk_face_dataset.test_age[i]] += 1
    race_freq[utk_face_dataset.test_race[i]] += 1
    gend_freq[utk_face_dataset.test_gender[i]] += 1

print(utk_face_dataset.test_data[0])
plt.imshow(utk_face_dataset.test_data[0])

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
    "attr_dom_size": 5,            # The number of classes of that attribute
    "attr_imb_cls": [2],           # The specific attribute classes to imbalance
    "per_attr_imb_train": 1,      # The number of training data points to keep for each affected value of the imbalance attribute domain
    "per_attr_imb_val": 1,        # The number of validation data points to keep for each affected value of the imbalance attribute domain
    "per_attr_imb_lake": 1,       # The number of lake data points to keep for each affected value of the imbalance attribute domain
    "per_attr_train": 2,          # The number of training data points to keep for each unaffected value of the imbalance attribute domain
    "per_attr_val": 2,            # The number of validation data points to keep for each unaffected value of the imbalance attribute domain
    "per_attr_lake": 2,            # The number of lake data points to keep for each unaffected value of the imbalance attribute domain
    "age_bins": [3*(1+x) for x in range(100)]
    }

# We perform the attribute imbalance
X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, selected_classes, num_cls = load_dataset_custom(datadir, dset_name, feature, split_cfg, isnumpy, augVal)

age_freq = [0 for x in range(test_set.n_age)]
race_freq = [0 for x in range(test_set.n_race)]
gend_freq = [0 for x in range(test_set.n_gender)]
test_this_set = test_set
test_this_data = X_val
test_this_label = y_val
print("CLASSES:", num_cls)
for i in range(len(test_this_set)):
    
    #plt.imshow(test_this_data[i])
    #label = test_this_label[i]
    #print(F"{i+1} label: {label}")
    plt.show()
    age = test_this_set.test_age[i]
    race = test_this_set.test_race[i]
    gender = test_this_set.test_gender[i]
    #print(F"{i+1} (a,g,r): {age}, {gender}, {race}")
    age_freq[test_this_set.test_age[i]] += 1
    race_freq[test_this_set.test_race[i]] += 1
    gend_freq[test_this_set.test_gender[i]] += 1
    
print(age_freq)
print(race_freq)
print(gend_freq)

# Should break
print(test_set[3000])