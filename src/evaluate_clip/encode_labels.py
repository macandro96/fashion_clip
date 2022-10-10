import pandas as pd
import numpy as np
import open_clip
import torch
import pickle
import os
import argparse

def encode_text(model, text_inputs):
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    return text_features


def encode(args, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the label
    df = pd.read_csv('/vast/am10150/fashion_clip/data/raw/labels.csv')
    df = df.sort_values(by=['labelId'])

    if not args.with_prompt:
        labels = df['labelName'].values
        file_name = 'label_features_all_vitb32_oc.pkl'
    
    else:
        df['prompt'] = "the "+df['taskName']+" of dress is "+df['labelName']
        labels = df['prompt'].values
        file_name = 'label_features_prompt_all_vitb32_oc.pkl'

    text_inputs = torch.cat([open_clip.tokenize(c) for c in labels]).to(device)

    text_features = encode_text(model, text_inputs)
    
    dump_dir = '/vast/am10150/fashion_clip/data/processed'
    

    os.makedirs(dump_dir, exist_ok=True)

    with open(os.path.join(dump_dir, file_name), 'wb') as f:
        pickle.dump(text_features.cpu().numpy(), f)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--with_prompt', action='store_true')
    args = args.parse_args()
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    encode(args, model)
