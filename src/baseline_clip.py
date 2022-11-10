import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import open_clip
from  deepfashion_data import DeepFashionDataset
import tqdm
import torchvision.transforms as transforms
from sklearn.metrics import top_k_accuracy_score
import numpy as np


class BaselineCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        
        self.model = clip_model
        
    def forward(self, image, labels, prompt=''):
        image_features = self.model.encode_image(image)
        text_inputs = torch.cat([open_clip.tokenize(f"{prompt}{c}") for c in labels]).to(image.device)
        text_features = self.model.encode_text(text_inputs)

        return image_features, text_features

    def predict(self, image, labels):
        image_features, text_features = self.forward(image, labels)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T)
        return similarity


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    model = BaselineCLIP(model).to(device)
    train_dset = DeepFashionDataset(transforms_=preprocess, dataset_type='train')
    dataset = DeepFashionDataset(transforms_=preprocess, dataset_type='test')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=3)
    all_labels = train_dset.all_categories
    targets = []
    predictions = []
    with torch.no_grad():
        for image, label_idx, label in tqdm.tqdm(dataloader):
            image = image.to(device)
            label_idx = label_idx.to(device)
            targets.append(label_idx.cpu().numpy())
            
            similarity = model.predict(image, all_labels)
            predictions.append(similarity.cpu().numpy())

    targets = np.concatenate(targets)
    predictions = np.concatenate(predictions)
    for k in range(1, 6):
        print(f"top-{k} accuracy: {top_k_accuracy_score(targets, predictions, k=k)}")

if __name__ == "__main__":
    main()
