import yaml
from models import CustomCLIPWrapper
import clip
import torch

CHECKPOINT = '/vast/sk8974/experiments/cv_proj/scripts/train-CLIP-FT/lightning_logs/version_27098889/checkpoints/epoch=25-step=526760.ckpt'
DEVICE = 'cuda'

clp, preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)

for p in clp.parameters(): 
    p.data = p.data.float() 
    if p.grad:
        p.grad.data = p.grad.data.float()

model = CustomCLIPWrapper(clp.visual, clp.transformer, 32, avg_word_embs=True)

checkpoint = torch.load(CHECKPOINT)
model.load_state_dict(checkpoint["state_dict"])
import pdb; pdb.set_trace()
