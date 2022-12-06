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
    def __init__(self, root='/vast/am10150/fashion_clip/data/raw/deepfashion/dataset', transforms_=None, dataset_type='train', category_only=True):
        """
        Parameters
        ----------
        root: the root of the DeepFashion dataset. This is the folder
          which contains the subdirectories 'Anno', 'Img', etc.
          It is assumed that in 'Img' the directory 'img_converted'
          exists, which gets created by running the script `resize.sh`.
        """
        self.transform = transforms_
        self.category_only = category_only

        assert dataset_type in ['train', 'val', 'test']
        self.dataset_type = dataset_type
        self.root = root

        # get image paths
        self.img_paths = self._get_dataset()

        # get all categories and image wise categories
        self.all_categories = self._get_all_categories()
        self.img_categories = self._img2category()

        # get all attributes
        if not self.category_only:
            self.all_attributes = self._get_all_attributes()
            self.img_attributes = self._get_img2attr()

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
        attr_file = os.path.join(self.root, "Anno_coarse", "list_attr_cloth.txt")
        f = open(attr_file, 'r')
        num_attrs = int(f.readline())
        f.readline()  # skip header

        all_attrs = []
        for line in f:
            attr = line.split()[0]
            all_attrs.append(attr)

        return all_attrs
    
    def _get_img2attr(self):
        img_attr_file = os.path.join(self.root, "Anno_coarse", "list_attr_img.txt")
        f = open(img_attr_file, 'r')
        num_imgs = int(f.readline())
        f.readline()

        img_attrs = {}
        for line in f:
            spl = line.split()
            img_path = spl[0]
            attrs = []
            for i, attr in enumerate(spl[1:]):
                if attr == '-1':
                    attrs.append(0)
                else:
                    attrs.append(int(attr))
            img_attrs[img_path] = attrs
        return img_attrs

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        
        img = Image.open(os.path.join(self.root, img_path))
        img = self.transform(img)
        category_label_idx = self.img_categories[img_path]

        category_idx = torch.Tensor([category_label_idx-1]).long()
        category_labels = self.all_categories[category_label_idx-1]
        if self.category_only:
            return img, category_idx, category_labels            # return img, category_label_idx, category_labels
        else:
            attribute_label_idx = self.img_attributes[img_path]
            attribute_label_idx = torch.Tensor([attribute_label_idx-1])             # attribute_label_idx = torch.Tensor([attribute_label_idx])
            return img, category_idx, category_labels, attribute_label_idx #return img, category_label_idx, category_labels, attribute_label_idx
        

    def __len__(self):
        return len(self.img_paths)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    ])
    ds = DeepFashionDataset(transforms_=train_transforms, category_only=False)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=15)
    max_label = 0
    attr_idx_sum = torch.zeros(1,1000)
    total_instances = 0
    for img, label_idx, label, attribute_idx in tqdm.tqdm(loader):
        attr_idx_sum += attribute_idx.sum(dim=0)
        total_instances += attribute_idx.shape[0]
        # print(label_idx, label)?
        max_label = max(max_label, label_idx.max())
        # print(img.shape, category.shape, attr.shape)
    weights = total_instances / attr_idx_sum
    import pdb; pdb.set_trace()
    weights.numpy().dump('attribute_class_weight.npy')
    print(max_label)