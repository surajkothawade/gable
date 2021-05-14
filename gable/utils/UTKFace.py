from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity
from typing import Callable, List, Optional, Union
from torch.utils.model_zoo import tqdm
import tarfile 
import os
import PIL
import numpy as np

class UTKFace(VisionDataset):
    
    """
    Manages loading and transforming the data for the UTKFace dataset. 
    Mimics the other VisionDataset classes available in torchvision.datasets.
    """
    
    base_folder = "utkface"
    
    # Note: File ID is obtained from the share link!
    file_list = [
        # File ID                       # md5   # filename
        ("0BxYys69jI14kYVM3aVhKS1VhRUk", None, "utkface.tar.gz")    
    ]
    
    def __init__(self,
            root: str,
            target_type: Union[List[str], str] = "age",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            age_bins = None
    ) -> None:
        
        # Call superconstructor
        super(UTKFace, self).__init__(root, transform=transform, target_transform=target_transform)
        
        # Download the dataset if needed
        if download:
            self.download()     
        
        self.target_attribute = target_type
        
        # Load dataset and populate in attributes
        self._get_datapoints_as_numpy(age_bins)
        
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
    
    def _get_unique_vals_in_array(self, array_to_check):
        
        checked = []
        unique_vals = 0
        for check_item in array_to_check:
            if check_item in checked:
                continue
            unique_vals = unique_vals + 1
            checked.append(check_item)
            
        return unique_vals        
        

    def _get_datapoints_as_numpy(self, age_bins):
                   
        # Get path of extracted dataset
        images_path = os.path.join(self.root, self.base_folder, "UTKFace")
        
        # Get all files
        all_image_files = os.listdir(images_path)
        
        # Go through each file name. If one represents a faulty attribute assignment, remove it.
        for image_file in all_image_files:
            
            image_attributes = image_file.split("_")
            
            # Delete image if it is not labeled correctly
            if len(image_attributes) != 4:
                os.remove(os.path.join(images_path, image_file))
                
        # Redo getting all files
        all_image_files = os.listdir(images_path)
        
        # Create numpy arrays to hold each datapoint
        age_attributes = np.zeros(len(all_image_files), dtype=np.int64)
        gender_attributes = np.zeros(len(all_image_files), dtype=np.int64)
        race_attributes = np.zeros(len(all_image_files), dtype=np.int64)
        images = np.zeros((len(all_image_files), 200, 200, 3), dtype=np.uint8)        

        assign_index = 0

        for i in range(len(all_image_files)):

            if i % (len(all_image_files) // 10) == 0:
                print(F"Loaded {i} of {len(all_image_files)} images")

            image_file_name = all_image_files[i]
            
            # Split the file name to elicit the attributes of the image
            image_attributes = image_file_name.split("_")
            
            # If an image is labeled in a faulty manner, skip it and delete the row
            if len(image_attributes) != 4:
                continue
            
            # Each file is named as "age_gender_race_datetime"
            age_attribute = int(image_attributes[0]) - 1 # Images are 1-indexed.
            gender_attribute = int(image_attributes[1])     
            race_attribute = int(image_attributes[2])

            # Ignore ages greater than 80
            if age_attribute > 65:
                continue

            # Bin the age if bins were provided
            if age_bins is not None:
                for j, age_cutoff in enumerate(age_bins):
                    if age_attribute < age_cutoff:
                        age_attribute = j
                        break

            # Populate labels
            age_attributes[assign_index] = age_attribute
            gender_attributes[assign_index] = gender_attribute
            race_attributes[assign_index] = race_attribute
            
            # Lastly, load the image and convert to numpy array
            image_path = os.path.join(images_path, image_file_name)
            image = PIL.Image.open(image_path)
            image_numpy = np.array(image)
            images[assign_index] = image_numpy
            
            assign_index += 1
        
        keep_indices = [x for x in range(assign_index)]
        
        # Shave off indices that were saved for those that weren't included by age
        age_attributes = age_attributes[keep_indices]
        race_attributes = race_attributes[keep_indices]
        gender_attributes = gender_attributes[keep_indices]
        images = images[keep_indices]
        
        print("IMAGES SIZE:", images.shape)
        
        # Set arrays as attributes of class
        self.age = age_attributes
        self.gender = gender_attributes
        self.race = race_attributes
        self.data = images
        
        self.n_age = self._get_unique_vals_in_array(self.age)
        self.n_gender = self._get_unique_vals_in_array(self.gender)
        self.n_race = self._get_unique_vals_in_array(self.race)
        
        # Set target array
        # We choose the age as the label of the image. If the 
        if self.target_attribute == "age":
            self.targets = self.age
            self.nclasses = self.n_age
        elif self.target_attribute == "gender":
            self.targets = self.gender
            self.nclasses = self.n_gender
        elif self.target_attribute == "race":
            self.targets = self.race
            self.nclasses = self.n_race

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