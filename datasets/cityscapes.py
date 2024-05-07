import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    # Define all the classes specified in the mapping
    classes = [
        CityscapesClass('background', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('road', 1, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('potholes', 2, 1, 'flat', 1, False, False, (165, 42, 42)),
        CityscapesClass('shoulder', 3, 255, 'flat', 1, False, True, (244, 35, 232)),
        CityscapesClass('vegetation', 4, 2, 'nature', 2, False, False, (0, 128, 0)),
        CityscapesClass('building', 5, 3, 'construction', 3, True, False, (255, 255, 0)),
        CityscapesClass('sky', 6, 4, 'sky', 4, False, False, (0, 0, 255)),
        CityscapesClass('animal', 7, 255, 'animal', 5, True, True, (220, 20, 60)),
        CityscapesClass('person', 8, 255, 'human', 6, True, True, (220, 20, 60)),
        CityscapesClass('vehicle', 9, 5, 'vehicle', 7, True, False, (255, 128, 64)),
        CityscapesClass('water body', 10, 255, 'void', 0, False, True, (51, 102, 255)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)

    id_to_train_id = np.array([c.train_id for c in classes])
    # id_to_train_id = {c.id: idx for idx, c in enumerate(classes)}

    def __init__(self, root, split='train', mode='fine', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.transform = transform
        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            if city == '.DS_Store':
                continue
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], self._get_target_suffix(self.mode, 'semantic'))
                self.targets.append(os.path.join(target_dir, target_name))


    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 6
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]
    

    # @classmethod
    # def encode_target(cls, target):
    #     # Map Class ID's to train IDs
    #     map = {
    #         0: 0, 7: 1, 5: 2, 8: 3, 21: 4, 11: 5, 23: 6, 24: 7, 26: 8, 4: 0
    #     }
    #     return np.vectorize(map.get)(target)

    # @classmethod
    # def decode_target(cls, target):
    #     # Map train IDs to colors
    #     colors = [cls.train_id_to_color[train_id] for train_id in np.nditer(target)]
        
    #     return np.array(colors).reshape(target.shape + (3,))


    def __getitem__(self, index):
        image_path = self.images[index]
        target_path = self.targets[index]
        
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(np.array(target))
        
        return image, target, image_path

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)
