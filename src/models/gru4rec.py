# src/models/gru4rec.py
import torch
import torch.nn as nn

def _init_gru(gru: nn.GRU):
    # Xavier for weights, zeros for bias
    for name, p in gru.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(p)
        elif 'bias' in name:
            nn.init.zeros_(p)

class Model(nn.Module):
    def __init__(self, cfg, n_items):
        super().__init__()
        self.max_len = int(cfg.get('max_len', 100))
        d = int(cfg.get('dim', 128))
        dropout = float(cfg.get('dropout', 0.2))

        self.item_emb = nn.Embedding(n_items, d, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()  # PAD는 0으로 고정

        self.gru = nn.GRU(d, d, num_layers=1, batch_first=True)
        _init_gru(self.gru)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

        self.out = nn.Linear(d, n_items, bias=False)
        # weight tying
        self.out.weight = self.item_emb.weight

    def forward(self, x, attn_mask):
        # x: (B,L) long, attn_mask: (B,L) {0,1}
        emb = self.item_emb(x)
        h, _ = self.gru(emb)
        h = self.dropout(h)
        h = self.norm(h)

        last_idx = attn_mask.sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        hT = h[torch.arange(x.size(0), device=x.device), last_idx]
        return self.out(hT)

    @torch.no_grad()
    def full_scores(self, x, attn_mask):
        return self.forward(x, attn_mask)
