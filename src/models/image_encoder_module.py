import torch
import torch.nn as nn
import pytorch_lightning as pl
import clip

class ImageEncoderModule(pl.LightningModule):
    def __init__(self, 
        model_name='ViT-B/32',
        lr=1e-4,
        weight_decay=1e-4
        ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.linear = torch.nn.Linear(512, 228)  # linear probe
        self.loss_fn = torch.nn.BCELoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.model.encode_image(x)
        x = torch.sigmoid(self.linear(x))
        return x

    def training_step(self, batch, batch_idx):
        image, text, label_ids = batch
        y_hat = self(image)
        loss = self.loss_fn(y_hat, label_ids)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, text, label_ids = batch
        y_hat = self(image)
        loss = self.loss_fn(y_hat, label_ids)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        param_list = list(self.model.visual.parameters()) + list(self.linear.parameters())
        optimizer = torch.optim.AdamW(param_list, lr=self.lr, weight_decay=self.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return optimizer

