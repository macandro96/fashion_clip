"""
Adapted from Christopher Beckham's code:
https://github.com/christopher-beckham/deepfashion_data_loader/blob/master/dataset.py
"""

import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import tqdm

class DeepFashionDataset(Dataset):
    def __init__(self, root='/vast/am10150/fashion_clip/data/raw/deepfashion/dataset', transforms_=None, dataset_type='train'):
        """
        Parameters
        ----------
        root: the root of the DeepFashion dataset. This is the folder
          which contains the subdirectories 'Anno', 'Img', etc.
          It is assumed that in 'Img' the directory 'img_converted'
          exists, which gets created by running the script `resize.sh`.
        """
        self.transform = transforms_
        assert dataset_type in ['train', 'val', 'test']
        self.dataset_type = dataset_type
        self.root = root

        # get image paths
        self.img_paths = self._get_dataset()

        # get all categories and image wise categories
        self.all_categories = self._get_all_categories()
        self.img_categories = self._img2category()

    def _get_dataset(self):
        split_file = os.path.join(self.root, "Eval/list_eval_partition.txt")
        f = open(split_file, 'r')
        num_imgs = int(f.readline())
        f.readline() # skip header
        
        img_paths = []
        for line in f:
            img_path, split = line.split()
            if split == self.dataset_type:
                img_paths.append(img_path)

        return img_paths

    def _get_all_categories(self):
        category_file = os.path.join(self.root, "Anno_coarse/list_category_cloth.txt")
        f = open(category_file, 'r')
        num_categories = int(f.readline())
        f.readline()
        
        all_categories = []
        for line in f:
            category = line.split()[0]
            all_categories.append(category)
        return all_categories
        
    def _img2category(self):
        img_category_file = os.path.join(self.root, "Anno_coarse/list_category_img.txt")
        f = open(img_category_file, 'r')
        num_imgs = int(f.readline())
        f.readline()

        img_categories = {}
        for line in f:
            img_path, category = line.split()
            img_categories[img_path] = int(category)
        
        return img_categories

    def _get_all_attributes(self):
        attr_file = os.path.join(self.root, "Anno_fine", "list_attr_cloth.txt")
        f = open(attr_file, 'r')
        num_attrs = int(f.readline())
        f.readline()

        all_attrs = []
        for line in f:
            attr = line.split()[0]
            all_attrs.append(attr)

        return all_attrs

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        
        img = Image.open(os.path.join(self.root, img_path))
        img = self.transform(img)
        label_idx = self.img_categories[img_path]
        return img, torch.Tensor([label_idx-1]).long(), self.all_categories[label_idx-1]
        

    def __len__(self):
        return len(self.img_paths)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    train_transforms = [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    ]
    ds = DeepFashionDataset(transforms_=train_transforms)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
    max_label = 0
    for img, label_idx, label in tqdm.tqdm(loader):
        # print(label_idx, label)?
        max_label = max(max_label, label_idx.max())
        # print(img.shape, category.shape, attr.shape)
    print(max_label)