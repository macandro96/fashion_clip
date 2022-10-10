import torch
import torchvision
import os
import numpy as np
import json
import pandas as pd
import torch.utils.data as data
import tqdm
import pickle
import torchvision
import open_clip


class iMaterialistDataset(data.Dataset):
    def __init__(self, data_type, transforms=None):
        super().__init__()
        self.data_type = data_type
        assert data_type in ['train', 'validation', 'test']

        # image data info
        self.data_dir = '/vast/am10150/fashion_clip/data/raw'
        self.image_dir = os.path.join(self.data_dir, data_type)
        self.image_files = [os.path.join(self.image_dir, image_name) for image_name in sorted(os.listdir(self.image_dir))]

        # load label info
        f = open(f'/vast/am10150/fashion_clip/data/raw/{data_type}.json', 'r')
        label_data = json.load(f)

        self.annotations = label_data['annotations']

        # id -> label info
        df = pd.read_csv('/vast/am10150/fashion_clip/data/raw/labels.csv')
        df = df.sort_values(by=['labelId'])
        self.id2labels = dict(zip(df.labelId, zip(df.labelName, df.taskName)))
        self.id2tokens = dict(zip(df.labelId, open_clip.tokenize(df.labelName)))
        self.transforms = transforms
        # import pdb; pdb.set_trace()

    def __getitem__(self, index):
        
        image = torchvision.io.read_image(self.image_files[index])
        image_id = int(self.image_files[index].split('/')[-1].split('.')[0])
    

        annotation = self.annotations[int(image_id)-1]
        assert int(annotation['imageId']) == image_id

        label_ids = annotation['labelId']
        found_labels = []
        for idx in range(len(label_ids)):
            label_ids[idx] = int(label_ids[idx])
            label_id = label_ids[idx]
            assert label_id in range(229)

            found_labels.append(self.id2labels[label_id])
        
        if self.transforms is not None:
            image = self.transforms(image)
        label_ids = torch.Tensor(label_ids).long()
        label_tensor = torch.zeros(228, dtype=torch.long)
        label_tensor[label_ids - 1] = 1
        return image, label_tensor
    
    def __len__(self):
        return len(self.image_files)

if __name__ == '__main__':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224))
    ])
    dataset = iMaterialistDataset('train', transforms=transforms)
    
    dataloader = data.DataLoader(dataset, batch_size=128, num_workers=32)

    max_height, max_width = 0, 0
    for i, (images, labels) in tqdm.tqdm(enumerate(dataloader)):
        max_height = max(max_height, images.shape[2])
        max_width = max(max_width, images.shape[3])

    print(max_height, max_width)

    



    

