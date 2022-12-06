

from models_fine import CustomCLIPWrapper

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
from sklearn.metrics import top_k_accuracy_score, f1_score
import numpy as np
import clip

import yaml



class BaselineCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        
        self.model = clip_model
        
    def forward(self, image, labels, prompt=''):
        image_features = self.model.encode_image(image)
        text_inputs = torch.cat([clip.tokenize(f"{prompt}{c}") for c in labels]).to(image.device)
        text_features = self.model.encode_text(text_inputs)

        return image_features, text_features

    def predict(self, image, labels):
        image_features, text_features = self.forward(image, labels, prompt='')
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T)
        return similarity


def main():
    baseline = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if baseline:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model = BaselineCLIP(model).to(device)
        
    else:
        # CHECKPOINT = '/vast/sk8974/experiments/cv_proj/scripts/train-CLIP-FT/lightning_logs/version_27530617/checkpoints/epoch=31-step=507232.ckpt'
        # CHECKPOINT ='/vast/sk8974/experiments/cv_proj/scripts/train-CLIP-FT/lightning_logs/version_27487456/checkpoints/epoch=31-step=507232.ckpt'
        CHECKPOINT = '/vast/sk8974/experiments/cv_proj/scripts/train-CLIP-FT/lightning_logs/version_27487456/checkpoints/epoch=8-step=142659.ckpt'
        DEVICE = 'cuda'

        clp, preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)

        for p in clp.parameters(): 
            p.data = p.data.float() 
            if p.grad:
                p.grad.data = p.grad.data.float()

        model = CustomCLIPWrapper(clp.visual, clp.transformer, 32, avg_word_embs=True)
        checkpoint = torch.load(CHECKPOINT)
        model.load_state_dict(checkpoint["state_dict"])
        
        # model, preprocess = clip.load(model_path, device='cpu', jit=False)
        model = BaselineCLIP(model.model).to(device)
        
    train_dset = DeepFashionDataset(transforms_=preprocess, dataset_type='train')
    dataset = DeepFashionDataset(transforms_=preprocess, dataset_type='test')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
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

    pred_top1 = np.argmax(predictions,axis=1)
    
    for k in range(1, 6):
        print(f"top-{k} accuracy: {top_k_accuracy_score(targets, predictions, k=k,labels=list(range(50)))}")

    f1 = f1_score(targets, pred_top1, average='weighted')

    print(f" \n f1 score: {f1}")

if __name__ == "__main__":
    main()
