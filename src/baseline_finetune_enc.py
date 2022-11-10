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
    def __init__(self, clip_model, num_classes=50):
        super().__init__()
        
        self.model = clip_model.visual

        self.linear = nn.Linear(512, num_classes)
        
    def forward(self, image):
        image_features = self.model(image)
        return self.linear(image_features)


class CLIPImageEncoderModule(pl.LightningModule):
    def __init__(self, clip_model, num_classes=50, lr=1e-7):
        super().__init__()
        self.lr = lr

        self.model = CLIPImageEncoder(clip_model, num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, image):
        return self.model(image)
    
    def training_step(self, batch, batch_idx):
        outs = self._common_step(batch, batch_idx)
        self.log('train_loss', outs['loss'])
        return outs['loss']
    
    def _common_step(self, batch, batch_idx):
        image, label, _ = batch
        label = label.squeeze()
        logits = self(image)
        loss = self.loss_fn(logits, label)
        preds = logits.softmax(dim=-1)

        return {
            'loss': loss,
            'preds': preds,
            'labels': label,
        }

    def validation_step(self, batch, batch_idx):
        outs = self._common_step(batch, batch_idx)
        self.log('val_loss', outs['loss'])
        acc = top_k_accuracy_score(outs['labels'].cpu().numpy(), outs['preds'].cpu().numpy(), k=1, labels=list(range(50)))
        self.log('val_acc', acc)
        return outs
    
    def _get_metrics(self, logs):
        preds, labels = [], []
        for log in logs:
            preds.append(log['preds'].cpu().numpy())
            labels.append(log['labels'].cpu().numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        
        return {
            'acc1': top_k_accuracy_score(labels, preds, k=1, labels=list(range(50))),
            'acc2': top_k_accuracy_score(labels, preds, k=2, labels=list(range(50))),
            'acc3': top_k_accuracy_score(labels, preds, k=3, labels=list(range(50))),
            'acc4': top_k_accuracy_score(labels, preds, k=4, labels=list(range(50))),
            'acc5': top_k_accuracy_score(labels, preds, k=5, labels=list(range(50)))
        }

    def validation_epoch_end(self, val_logs):
        accuracy = self._get_metrics(val_logs)['acc1']
        print(accuracy)
        self.log('val_acc', accuracy, prog_bar=True)
    
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

    train_dataset = DeepFashionDataset(transforms_=preprocess, dataset_type='train')
    val_dataset = DeepFashionDataset(transforms_=preprocess, dataset_type='val')
    test_dataset = DeepFashionDataset(transforms_=preprocess, dataset_type='test')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=15)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=15)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=15)

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
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=20, callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(model, train_dataloader, test_dataloader)
    trainer.test(model, test_dataloader)

if __name__ == '__main__':
    model_name = 'ViT-B-32-quickgelu'
    pretrained = 'laion400m_e32'
    main(model_name, pretrained)

    

    