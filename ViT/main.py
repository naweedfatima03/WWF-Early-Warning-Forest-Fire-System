import torch
from torch import nn
from d2l import torch as d2l
import Transformer, Trainer
from Transformer import ViT
from Trainer import Trainer

img_size, patch_size = 28, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr)
trainer = Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
train_dataloader = data.train_dataloader()
val_dataloader = data.val_dataloader()
print('starting training')
# trainer.fit(model, data)