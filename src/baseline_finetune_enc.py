import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


class CLIPImageEncoder(nn.Module):
    def __init__(self, clip_model, num_categories=50, num_attributes=1000):
        super().__init__()
        
        self.model = clip_model.visual

        self.linear_categories = nn.Linear(512, num_categories)
        self.linear_attributes = nn.Linear(512, num_attributes)
        
    def forward(self, image):
        image_features = self.model(image)
        categories_logits = self.linear_categories(image_features)
        attributes_logits = self.linear_attributes(image_features)
        return categories_logits, attributes_logits


class CLIPImageEncoderModule(pl.LightningModule):
    def __init__(self, clip_model, num_categories=50, num_attributes=1000, lr=1e-7, pos_weight_file='attribute_class_weight.npy'):
        super().__init__()
        self.lr = lr
        pos_weights = torch.from_numpy(np.load(pos_weight_file, allow_pickle=True))
        self.model = CLIPImageEncoder(clip_model, num_categories=num_categories, num_attributes=num_attributes)
        self.category_loss = nn.CrossEntropyLoss()
        self.attribute_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    def forward(self, image):
        return self.model(image)
    
    def training_step(self, batch, batch_idx):
        outs = self._common_step(batch, batch_idx)
        self.log('train_loss', outs['loss'])
        return outs['loss']
    
    def _common_step(self, batch, batch_idx):
        image, categories, _, attributes = batch
        categories = categories.squeeze()
        attributes = attributes.squeeze(1)
        cat_logits, attr_logits = self(image)
        
        # category_loss = self.category_loss(cat_logits, categories)
        attribute_loss = self.attribute_loss(attr_logits, attributes)

        cat_preds = cat_logits.detach().softmax(dim=-1)

        return {
            'loss': attribute_loss,
            'cat_preds': cat_preds,
            'attr_preds': attr_logits,
            'categories': categories,
            'attributes': attributes
        }

    def _get_recall_example(self, indices, target):
        tp, fn = 0, 0
        for i, t in enumerate(target):
            if t == 1:
                if i in indices:
                    tp += 1
                else:
                    fn += 1
        if tp + fn == 0:
            return -1
        recall = tp / (tp + fn)
        return recall

    def _get_topk_recall(self, targets, preds, top_k=1):
        indices = preds.argsort(axis=-1)[:,::-1][:, :top_k]
        recall = []
        empty = 0
        for i in range(preds.shape[0]):
            recall_i = self._get_recall_example(indices[i], targets[i])
            if recall_i != -1:
                recall.append(recall_i)
            else:
                empty += 1
        sorted_recall = sorted(recall)[::-1]
        calc_recall = sum(sorted_recall[:50]) / min(50, len(sorted_recall)-empty)
        return calc_recall

    def validation_step(self, batch, batch_idx):
        outs = self._common_step(batch, batch_idx)
        self.log('val_loss', outs['loss'])
        cat_acc = top_k_accuracy_score(outs['categories'].cpu().numpy(), outs['cat_preds'].cpu().numpy(), k=1, labels=list(range(50)))
        self._get_topk_recall(outs['attributes'].cpu().numpy(), outs['attr_preds'].cpu().numpy(), top_k=3)
        self.log('val_acc', cat_acc)
        return outs
    
    def _get_metrics(self, logs):
        cat_preds, cat_labels = [], []
        attr_preds, attr_labels = [], []
        for log in logs:
            cat_preds.append(log['cat_preds'].cpu().numpy())
            cat_labels.append(log['categories'].cpu().numpy())

            attr_preds.append(log['attr_preds'].cpu().numpy())
            attr_labels.append(log['attributes'].cpu().numpy())
        
        cat_preds = np.concatenate(cat_preds)
        cat_labels = np.concatenate(cat_labels)

        attr_preds = np.concatenate(attr_preds)
        attr_labels = np.concatenate(attr_labels)

        
        return {
            'cat_acc1': top_k_accuracy_score(cat_labels, cat_preds, k=1, labels=list(range(50))),
            'cat_acc2': top_k_accuracy_score(cat_labels, cat_preds, k=2, labels=list(range(50))),
            'cat_acc3': top_k_accuracy_score(cat_labels, cat_preds, k=3, labels=list(range(50))),
            'cat_acc4': top_k_accuracy_score(cat_labels, cat_preds, k=4, labels=list(range(50))),
            'cat_acc5': top_k_accuracy_score(cat_labels, cat_preds, k=5, labels=list(range(50))),

            'attr_recall3': self._get_topk_recall(attr_labels, attr_preds, top_k=3),
            'attr_recall5': self._get_topk_recall(attr_labels, attr_preds, top_k=5),
        }

    def validation_epoch_end(self, val_logs):
        cat_accuracy = self._get_metrics(val_logs)['cat_acc1']
        attr_recall3 = self._get_metrics(val_logs)['attr_recall3']
        print(cat_accuracy)
        print(attr_recall3)
        self.log('val_acc', cat_accuracy, prog_bar=True)
        self.log('val_attr_recall3', attr_recall3, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)
    
    def test_epoch_end(self, test_logs):
        accuracy = self._get_metrics(test_logs)
        print(f"Test accuracies {accuracy}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    

def main(model_name, pretrained):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = CLIPImageEncoderModule(model).to(device)

    train_dataset = DeepFashionDataset(transforms_=preprocess, dataset_type='train', category_only=False)
    val_dataset = DeepFashionDataset(transforms_=preprocess, dataset_type='val', category_only=False)
    test_dataset = DeepFashionDataset(transforms_=preprocess, dataset_type='test', category_only=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=3)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=3)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=3)

    model_dir = os.path.join('img_enc_finetune', model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=model_dir,
        filename='best_model',
        save_top_k=1,
        mode='min',
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min")
    if device=='cuda':
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=20, callbacks=[early_stop_callback, checkpoint_callback])
    else:
        trainer = pl.Trainer(accelerator='cpu', max_epochs=20, callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(model, train_dataloader, test_dataloader)
    trainer.test(model, test_dataloader)

if __name__ == '__main__':
    model_name = 'ViT-B-32-quickgelu'
    pretrained = 'laion400m_e32'
    main(model_name, pretrained)

    

    
