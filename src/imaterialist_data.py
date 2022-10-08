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

class iMaterialistDataset(data.Dataset):
    def __init__(self, data_type, transforms=None):
        super().__init__()
        self.data_type = data_type
        assert data_type in ['train', 'validation', 'test']
        self.invalid = []

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
        self.transforms = transforms


    def __getitem__(self, index):
        try:
            image = torchvision.io.read_image(self.image_files[index])
            image_id = int(self.image_files[index].split('/')[-1].split('.')[0])
        except:
            print(self.image_files[index])
            self.invalid.append(self.image_files[index])
            return torch.randn(3, 224, 224)
            # raise

        annotation = self.annotations[int(image_id)-1]
        assert int(annotation['imageId']) == image_id

        label_ids = annotation['labelId']
        found_labels = []
        for label_id in label_ids:
            label_id = int(label_id)
            assert label_id in range(229)

            found_labels.append(self.id2labels[label_id])
        
        if self.transforms is not None:
            image = self.transforms(image)
        # print(label_ids)
        return image
    
    def __len__(self):
        return len(self.image_files)

if __name__ == '__main__':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224))
    ])
    dataset = iMaterialistDataset('train', transforms=transforms)
    
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=3)

    max_height, max_width = 0, 0
    for i, images in tqdm.tqdm(enumerate(dataloader)):
        max_height = max(max_height, images.shape[2])
        max_width = max(max_width, images.shape[3])
        # print(images.shape)
        # print(labels)
        # if i == 3:
        #     break
    print(max_height, max_width)
    print(len(dataset.invalid))
    pickle.dump(dataset.invalid, open('invalid.pkl', 'wb'))

    



    

