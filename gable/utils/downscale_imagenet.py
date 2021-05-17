from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity
from typing import Callable, List, Optional, Union
from torch.utils.model_zoo import tqdm
import tarfile 
import os
import PIL
import numpy as np

class ImageNet_Downscale(VisionDataset):
    
    """
    Manages loading and transforming the data for the downscaled ImageNet dataset. 
    Mimics the other VisionDataset classes available in torchvision.datasets.
    """
    
    base_folder = "downscale_imagenet"
    
    def __init__(self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        
        # Call superconstructor
        super(ImageNet_Downscale, self).__init__(root, transform=transform, target_transform=target_transform)
        
        # Load dataset and populate in attributes
        self._get_datapoints_as_numpy()
        
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

    def _get_datapoints_as_numpy(self):
                   
        # Get path of train npz batches
        train_images_path = os.path.join(self.root, self.base_folder, "Imagenet32_train_npz")
        
        # Get all batches
        image_batch_files = os.listdir(train_images_path)
        
        numpy_train_data = None
        
        # Go through each file. For the loaded numpy array, concatenate it to a running total
        for image_batch_file in image_batch_files:
                        
            if numpy_train_data is None:
                numpy_train_data = np.load(image_batch_file)
            else:
                numpy_train_data = np.concatenate(numpy_train_data, np.load(image_batch_file))
        
        # Repeat for test data.
        # Get path of train npz batches
        test_images_path = os.path.join(self.root, self.base_folder, "Imagenet32_val_npz")
        
        # Get all batches
        image_batch_files = os.listdir(test_images_path)
        
        numpy_test_data = None
        
        # Go through each file. For the loaded numpy array, concatenate it to a running total
        for image_batch_file in image_batch_files:
                        
            if numpy_test_data is None:
                numpy_test_data = np.load(image_batch_file)
            else:
                numpy_test_data = np.concatenate(numpy_test_data, np.load(image_batch_file))
                
        # Set attributes in this Dataset.
        print(numpy_train_data.shape)