!pip install einops
!pip install tensorflow-datasets

import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import math
import os
from torchvision.datasets import ImageFolder
import torchvision
from torchvision import transforms
import tensorflow as tf
import torch
import tensorflow_datasets as tfds
from torch.utils.data import DataLoader
import time
import pickle
from torch.utils.data import random_split
import numpy as np

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size : int=224):
      self.patch_size = patch_size
      super().__init__()
      self.projection = nn.Sequential(
          nn.Conv2d(in_channels, emb_size, kernel_size = patch_size, stride = patch_size),
          Rearrange('b e (h) (w) -> b (h w) e')
        )
      self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))

    def positional_encoding(self, x: Tensor) -> Tensor:

      b,max_len,d_model = x.shape

      pe = torch.zeros(max_len, d_model)
      position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

      div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

      pe[:, 0::2] = torch.sin(position * div_term)
      pe[:, 1::2] = torch.cos(position * div_term)
      pe = pe.unsqueeze(0)
      pe = repeat(pe,'() t d -> b t d', b=b)

      return x + pe

    def forward(self, x: Tensor) -> Tensor:
      b,_,_,_ = x.shape
      x = self.projection(x)
      cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
      x = torch.cat([cls_token, x], dim=1)
      x = self.positional_encoding(x)

      return x

class EfficientHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), 'b n (h d qkv) -> (qkv) b h n d', h = self.num_heads, qkv = 3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        attn_values = []

        for i in range(self.num_heads):
            q = F.softmax(queries[:,i,:,:], dim = 2)
            k = F.softmax(keys[:,i,:,:], dim = 1)
            v = values[:,i,:,:]

            context = torch.einsum('bnk, bnv -> bkv', k, v)

            attn_value = (
                torch.einsum('bnl, blv -> bnv', q, context)
            )

            attn_values.append(attn_value)

        aggregated_values = torch.cat(attn_values, dim=2)
        out = self.projection(aggregated_values)

        return out

  class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

  class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

  class ETransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                EfficientHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

  class ETransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[ETransformerEncoderBlock(**kwargs) for _ in range(depth)])

  class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))

  class EViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 224,
                 depth: int = 12,
                 n_classes: int = 1000,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            ETransformerEncoder(depth, emb_size = emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

  
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  
  
