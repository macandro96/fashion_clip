import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from PIL import Image
import matplotlib.image as mpimg


def get_label(data, image_id, id2labels):

    f = open(f'/vast/am10150/fashion_clip/data/raw/{data}.json', 'r')
    data = json.load(f)

    annotations = data['annotations']
    
    annotation = annotations[int(image_id)-1]
    print(annotation)
    assert int(annotation['imageId']) == image_id

    label_ids = annotation['labelId']
    found_labels = []
    for label_id in label_ids:
        label_id = int(label_id)
        assert label_id in range(229)

        found_labels.append(id2labels[label_id])
    
    return found_labels


def get_image(image_id=None, data=None):
    if image_id is None or data is None:
        data = np.random.choice(['train', 'validation'])
        imgs = os.listdir(f'/vast/am10150/fashion_clip/data/raw/{data}')
        image_id = np.random.randint(0, len(imgs)-1)
        image_id = imgs[image_id]
        
        image_id = int(image_id.split('/')[-1].split('.')[0])
    
    else:
        assert data in ['train', 'validation']
    
    dir = os.path.join('/vast/am10150/fashion_clip/data/raw', data)
    image_path = os.path.join(dir, f'{image_id}.png')
    print(data, image_path)

    # img = mpimg.imread(image_path)
    # imgplot = plt.imshow(img)
    # plt.show()

    return data, image_id

def get_id2labels():
    df = pd.read_csv('/vast/am10150/fashion_clip/data/raw/labels.csv')
    df = df.sort_values(by=['labelId'])
    id2labels = dict(zip(df.labelId, zip(df.labelName, df.taskName)))

    return id2labels

def load_image_labels(data=None, image_id=None):
    id2labels = get_id2labels()
    data, image_id = get_image(image_id, data)
    labels = get_label(data, image_id, id2labels)
    print(labels)

    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--image_id', type=int, default=None)
    args.add_argument('--data', type=str, default=None)
    args = args.parse_args()

    load_image_labels(args.data, args.image_id)
    




