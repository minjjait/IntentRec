from .base import BaseRecModel
import torch
class Model(BaseRecModel):
    def __init__(self, cfg, n_items, pop_counts):
        super().__init__(); self.n_items = n_items
        pop = torch.zeros(n_items)
        for idx, cnt in pop_counts.items():
            if idx < n_items: pop[idx] = cnt
        self.register_buffer('scores_static', pop)
    def loss(self, batch): return torch.tensor(0.0, requires_grad=True)
    @torch.no_grad()
    def full_scores(self, x, attn_mask):
        s = self.scores_static.clone(); s[0] = -1e9; return s.unsqueeze(0)
