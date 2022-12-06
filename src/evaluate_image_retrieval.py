import torch
import numpy as np
from tqdm import tqdm
import open_clip
from deepfashion_data import DeepFashionDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models_fine import CustomCLIPWrapper
import clip

def evaluate(model, dataloader, all_labels, topk=5, device='cuda'):
    """
        The asumption is that the dataloader gives out (image, label_tensor, label_texts)
        model should be a clip model with encode_image and encode_text methods
    """
    # get image representations
    image_representations = []
    label_tensors = []
    model.eval()
    with torch.no_grad():
        for i, (image, label_tensor, _) in enumerate(tqdm(dataloader)):
            image = image.to(device)
            image_representations.append(model.encode_image(image))
            label_tensors.append(label_tensor.to(device))
            # break
        image_representations = torch.cat(image_representations, dim=0)
        label_tensors = torch.cat(label_tensors, dim=0) 

        # get label representations
        text_inputs = torch.cat([open_clip.tokenize(c) for c in all_labels]).to(device)
        label_representations = model.encode_text(text_inputs)

        # normalize image and label representations
        image_representations = F.normalize(image_representations, dim=1)
        label_representations = F.normalize(label_representations, dim=1)
    

    if label_tensors.shape[1] == 1:
        # need one hot encoding
        label_tensors = F.one_hot(label_tensors.squeeze(1), num_classes=len(all_labels))
        label_tensors = label_tensors.to(device)

    get_retrieval_metrics(image_representations, label_representations, label_tensors, topk=topk)


def get_retrieval_metrics(image_ft, label_ft, label_tensors, topk=5):
    pred = torch.matmul(label_ft, torch.transpose(image_ft, 1, 0))

    pred = pred.topk(topk, dim=1)[1]
    num_txt = pred.shape[0]

    prec, rec = [], []
    for i in range(num_txt):
        img_idxs = pred[i]
        tp = 0
        for img_idx in img_idxs:
            if label_tensors[img_idx][i] == 1:
                tp += 1
    
        if label_tensors[:, i].sum() == 0:
            continue
        prec.append(tp / topk)
        rec.append(tp / min(label_tensors[:, i].sum().cpu().numpy(), topk))
    
    
    precision, recall = np.mean(prec), np.mean(rec)
    hits_at_k = np.mean([p > 0 for p in prec])
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Precision: {np.mean(prec):.4f}, Recall: {np.mean(rec):.4f}, F1: {np.mean(f1):.4f}, Hits@{topk}: {hits_at_k:.4f}")
    return precision, recall, f1, hits_at_k

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32', device=device)

    baseline = False
    if baseline:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        # model = BaselineCLIP(model).to(device)

    else:
        # CHECKPOINT = '/vast/sk8974/experiments/cv_proj/scripts/train-CLIP-FT/lightning_logs/version_27530617/checkpoints/epoch=31-step=507232.ckpt'
        # CHECKPOINT = '/vast/sk8974/experiments/cv_proj/scripts/train-CLIP-FT/lightning_logs/version_27487456/checkpoints/epoch=31-step=507232.ckpt'
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
        model = model.model.to(device)
        # model = BaselineCLIP(model.model).to(device)

    deepfashion_test = DeepFashionDataset(transforms_=preprocess, category_only=True, dataset_type='test')
    
    test_dataloader = DataLoader(deepfashion_test, batch_size=128, shuffle=False, num_workers=14)
    evaluate(model, test_dataloader, deepfashion_test.all_categories, topk=5)

