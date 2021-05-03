from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity
from typing import Callable, List, Optional, Union
from torch.utils.model_zoo import tqdm
import tarfile 
import zipfile
import csv
import os
import PIL
import numpy as np

class FairFace(VisionDataset):
    
    """
    Manages loading and transforming the data for the UTKFace dataset. 
    Mimics the other VisionDataset classes available in torchvision.datasets.
    """
    
    base_folder = "fairface"
    
    # Note: File ID is obtained from the share link!
    file_list = [
        # File ID                       # md5   # filename
        ("1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86", None, "fairface.zip"),
        ("1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH", None, "train_labels.csv"),
        ("1wOdja-ezstMEp81tX1a-EYkFebev4h7D", None, "val_labels.csv")
    ]
    
    def __init__(self,
            root: str,
            target_type: Union[List[str], str] = "age",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            load_cap: int = 100000
    ) -> None:
        
        # Call superconstructor
        super(FairFace, self).__init__(root, transform=transform, target_transform=target_transform)
        
        # Download the dataset if needed
        if download:
            self.download()     
        
        self.target_attribute = target_type
        
        # Load dataset and populate in attributes
        self._get_datapoints_as_numpy(load_cap)
        
    def __getitem__(self, index):
        
        target = self.targets[index]
        
        # Conforming to other datasets
        image = PIL.Image.fromarray(self.data[index])
        
        # If a transform was provided, transform the image
        if self.transform is not None:
            image = self.transform(image)

        # If a target transform was provided, transform the target
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
        
    def __len__(self) -> int:
        
        # Return the length of one of the attribute arrays
        return len(self.age)

    def get_age_label(self, string_label):
        
        if string_label == "0-2":
            return 0
        elif string_label == "3-9":
            return 1
        elif string_label == "10-19":
            return 2
        elif string_label == "20-29":
            return 3
        elif string_label == "30-39":
            return 4
        elif string_label == "40-49":
            return 5
        elif string_label == "50-59":
            return 6
        elif string_label == "60-69":
            return 7
        else:
            return 8

    def get_gender_label(self, string_label):
        
        if string_label == "Male":
            return 0
        else:
            return 1
        
    def get_race_label(self, string_label):
        
        if string_label == "East Asian":
            return 0
        elif string_label == "SE Asian":
            return 1
        elif string_label == "Indian":
            return 2
        elif string_label == "Middle Eastern":
            return 3
        elif string_label == "Latino_Hispanic":
            return 4
        elif string_label == "White":
            return 5
        else:
            return 6

    def _get_datapoints_as_numpy(self, load_cap):
                   
        # Get path of extracted train dataset
        train_images_path = os.path.join(self.root, self.base_folder, "train")
        
        loaded_images = 0

        # Get all files
        all_image_files = os.listdir(train_images_path)
                
        train_to_load = min(load_cap, len(all_image_files))

        # Create numpy arrays to hold each train datapoint
        train_age_attributes = np.zeros(train_to_load, dtype=np.int64)
        train_gender_attributes = np.zeros(train_to_load, dtype=np.int64)
        train_race_attributes = np.zeros(train_to_load, dtype=np.int64)
        train_images = np.zeros((train_to_load, 224, 224, 3), dtype=np.uint8)        

        # Load all the training label attributes
        train_labels_path = os.path.join(self.root, self.base_folder, "train_labels.csv")
        
        print("Loading training data")
        
        with open(train_labels_path) as train_labels_csv:
            
            rowreader = csv.reader(train_labels_csv, delimiter=",")
            index = 0
            
            # Skip the label row
            rowreader.__next__()
            
            for (image_name, image_age, image_gender, image_race, _) in rowreader:
                
                if index >= train_to_load:
                    break

                image_path = os.path.join(self.root, self.base_folder, image_name)
                image = PIL.Image.open(image_path)
                image_numpy = np.array(image)
                train_images[index] = image_numpy
                
                train_age_attributes[index] = self.get_age_label(image_age)
                train_gender_attributes[index] = self.get_gender_label(image_gender)
                train_race_attributes[index] = self.get_race_label(image_race)
                index += 1

        loaded_images = train_to_load

        if loaded_images >= load_cap:
            # Set arrays as attributes of class
            self.age = train_age_attributes
            self.gender = train_gender_attributes
            self.race = train_race_attributes
            self.data = train_images
        
            # Set target array
            # We choose the age as the label of the image. If the 
            if self.target_attribute == "age":
                self.targets = self.age
            elif self.target_attribute == "gender":
                self.targets = self.gender
            elif self.target_attribute == "race":
                self.targets = self.race
            return

        # Get path of extracted val dataset
        val_images_path = os.path.join(self.root, self.base_folder, "val")
        
        # Get all files
        all_image_files = os.listdir(val_images_path)
                
        val_to_load = min(len(all_image_files), load_cap - loaded_images)

        # Create numpy arrays to hold each train datapoint
        val_age_attributes = np.zeros(val_to_load, dtype=np.int64)
        val_gender_attributes = np.zeros(val_to_load, dtype=np.int64)
        val_race_attributes = np.zeros(val_to_load, dtype=np.int64)
        val_images = np.zeros((val_to_load, 224, 224, 3), dtype=np.uint8)        

        # Load all the training label attributes
        val_labels_path = os.path.join(self.root, self.base_folder, "val_labels.csv")
        
        print("Loading validation data")
        
        with open(val_labels_path) as val_labels_csv:
            
            rowreader = csv.reader(val_labels_csv, delimiter=",")
            index = 0
            
            # Skip the label row
            rowreader.__next__()
            
            for (image_name, image_age, image_gender, image_race, _) in rowreader:
                
                if index >= val_to_load:
                    break
                    
                image_path = os.path.join(self.root, self.base_folder, image_name)
                image = PIL.Image.open(image_path)
                image_numpy = np.array(image)
                val_images[index] = image_numpy
                
                val_age_attributes[index] = self.get_age_label(image_age)
                val_gender_attributes[index] = self.get_gender_label(image_gender)
                val_race_attributes[index] = self.get_race_label(image_race)
                index += 1
        
        # Concatenate the training and validation information
        age_attributes = np.concatenate((train_age_attributes, val_age_attributes), axis=0)
        gender_attributes = np.concatenate((train_gender_attributes, val_gender_attributes), axis=0)
        race_attributes = np.concatenate((train_race_attributes, val_race_attributes), axis=0)
        images = np.concatenate((train_images, val_images), axis=0)
        
        # Set arrays as attributes of class
        self.age = age_attributes
        self.gender = gender_attributes
        self.race = race_attributes
        self.data = images
        
        # Set target array
        # We choose the age as the label of the image. If the 
        if self.target_attribute == "age":
            self.targets = self.age
        elif self.target_attribute == "gender":
            self.targets = self.gender
        elif self.target_attribute == "race":
            self.targets = self.race

    def _check_integrity(self) -> bool:
        
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            
            if not check_integrity(fpath, md5):
                return False
            
        return True
        
    def download(self) -> None:

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            self.download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)
            
            _, ext = os.path.splitext(filename)
            
            if ext in [".tar.gz", ".tar", ".gz"]:
                with tarfile.open(os.path.join(self.root, self.base_folder, filename), mode="r:gz") as f:
                    f.extractall(os.path.join(self.root, self.base_folder))
            elif ext in [".zip"]:
                with zipfile.ZipFile(os.path.join(self.root, self.base_folder, filename), mode="r") as f:
                    f.extractall(os.path.join(self.root, self.base_folder))

        print("Download passed")
        
        
    def download_file_from_google_drive(self, file_id: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None):
        
        import requests
        URL = "https://docs.google.com/uc?export=download"

        root = os.path.expanduser(root)
        if not filename:
            filename = file_id
        fpath = os.path.join(root, filename)

        os.makedirs(root, exist_ok=True)

        if os.path.isfile(fpath) and check_integrity(fpath, md5):
            print('Using downloaded and verified file: ' + fpath)
        else:
            session = requests.Session()

            response = session.get(URL, params = { 'id' : file_id }, stream = True)
            token = self.get_confirm_token(response)

            print(response.headers)
            print(len(response.content))

            if token:
                params = { 'id' : file_id, 'confirm' : token }
                response = session.get(URL, params = params, stream = True)
                print(response.headers)
                print(len(response.content))

            self.save_response_content(response, fpath)    

    def get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(self, response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            pbar = tqdm(total=None)
            progress = 0
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    progress += len(chunk)
                    pbar.update(progress - pbar.n)
            pbar.close()