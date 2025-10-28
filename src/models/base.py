import torch.nn as nn, torch
class BaseRecModel(nn.Module):
    def loss(self, batch): raise NotImplementedError
    @torch.no_grad()
    def full_scores(self, x, attn_mask): raise NotImplementedError
