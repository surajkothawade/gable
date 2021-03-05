import os
import torch
from torchvision.datasets.folder import has_file_allowed_extension, pil_loader, IMG_EXTENSIONS
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable, cast, Union, Any


def make_dataset(
    directory: str,
    classes: List[str],
    extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    limit_classes: Optional[Dict[str, int]] = None,
    shuffle: bool = False,
) -> List[Tuple[str, int]]:
    """Generates a list containing path to samples.

    Args:
        directory (str): root dataset directory.
        classes (List[str]): list of class names.
            This is basically the subfolder name which contains the images.
            In our case this is the attribute name. I am using "classes" variable name
            to make this sound generic.
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.
        limit_classes(optional, dict): Dictionary containing list of classes to limit.
            With key being the class name to limit and the value being the max allowed amount.
            Defaults to None.
        shuffle(bool): whether to shuffle the images present inside the subfolders. If False,
                       sorts the filenames present inside the folder


    Raises:
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.

    Returns:
        List[str]: [path_to_sample]
    """
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in classes:
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            limit_counter = 0
            if shuffle:
                np.random.shuffle(fnames)
            else:
                fname = sorted(fnames)
            for fname in fnames:
                if limit_classes:
                    if target_class in limit_classes:
                        if limit_counter < limit_classes[target_class]:
                            limit_counter += 1
                        else:
                            break
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    instances.append(path)
    return instances


def find_classes(folder: str) -> List[str]:
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        List: classes, where classes are relative to (dir).

    Ensures:
        No class is a subdirectory of another.
    """
    classes = [d.name for d in os.scandir(folder) if d.is_dir()]
    classes.sort()
    return classes


class AttributeDatasetFolder(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        attributes_to_include: Optional[List[str]] = None,
        attributes_to_limit: Optional[Dict[str, int]] = None,
        extensions: Union[Tuple[str, ...], None] = IMG_EXTENSIONS,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        loader: Callable = pil_loader,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shuffle_attributes: bool = False,
    ) -> None:
        """A dataloader which can be customized to limit some attributes.
           The data must be arranged in this way:
           root/class_name/attribute_name/image.png
           eg:
           root/dog/pug/image_1.png

        Args:
           root (string): Root directory path.
           attributes_to_include(list, optional): A list of attributes to include.
               If None, uses all the attributes present inside the class folder. Default None.
           attributes_to_limit(optional, dict): Dictionary containing list of classes to limit
               with key being the class name to limit and the value being the max allowed amount.
               Defaults to None.
           extensions (tuple[string]): A list of allowed extensions.
               both extensions and is_valid_file should not be passed.
           is_valid_file (callable, optional): A function that takes path of an Image file
               and check if the file is a valid file (used to check of corrupt files)
           loader (callable): A function to load an image given its path. Default: pil_loader
           transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``. Default None
           target_transform (callable, optional): A function/transform that takes in the
               target and transforms it. Default None
           shuffle_attributes(bool). Whether to shuffle the attributes. Useful while
                limiting a certain attribute. If False, sorts the attributes. Default False.

        Attributes:
           classes (list): List of the class names sorted alphabetically.
           class_to_idx (dict): Dict with items (class_name, class_index).
           samples (list): List of (sample path, class_index) tuples
           _attributes (list): List of all the attributes

        """
        self.root = root
        self.attributes_to_include = attributes_to_include
        self.attributes_to_limit = attributes_to_limit
        self.extensions = extensions
        self.is_valid_file = is_valid_file
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle_attributes = shuffle_attributes

        self._build_dataset()

    def _build_dataset(self):
        self.classes = find_classes(self.root)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self._samples_dict = {}
        self._attributes = []
        for cls_name in self.classes:
            attributes = find_classes(os.path.join(self.root, cls_name))
            if self.attributes_to_include:
                attributes = list(filter(lambda x: True if x in self.attributes_to_include else False, attributes))
            self._attributes.extend(attributes)
            self._samples_dict[cls_name] = make_dataset(
                directory=os.path.join(self.root, cls_name),
                classes=attributes,
                extensions=self.extensions,
                limit_classes=self.attributes_to_limit,
            )

        self.samples = []
        for cls_name, image_paths in self._samples_dict.items():
            targets = [self.class_to_idx[cls_name] for _ in range(len(image_paths))]
            self.samples.extend(zip(image_paths, targets))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
