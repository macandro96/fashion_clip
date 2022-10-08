import pandas as pd
import numpy as np
import clip
import torch
import pickle
import os
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import natsort
import tqdm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model, preprocess = clip.load('ViT-B/32', device=device)

    image_dir = '/vast/am10150/fashion_clip/data/raw/validation/'

    image_paths = os.listdir(image_dir)

    image_features_dict = {}
    with torch.no_grad():
        for image_file in tqdm.tqdm(image_paths):
            image_path = os.path.join(image_dir, image_file)
            image_id = int(image_file.split('.')[0])
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            image_features = model.encode_image(image)
            image_features_dict[image_id] = image_features.cpu().numpy()
    
    dump_dir = '/vast/am10150/fashion_clip/data/processed'
    os.makedirs(dump_dir, exist_ok=True)
    
    file_name = 'image_features_all.pkl'
    with open(os.path.join(dump_dir, file_name), 'wb') as f:
        pickle.dump(image_features_dict, f)


if __name__ == '__main__':
    main()