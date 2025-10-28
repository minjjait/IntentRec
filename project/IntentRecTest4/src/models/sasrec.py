import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self, cfg, n_items):
        super().__init__()
        self.max_len = cfg.get('max_len', 100); d = cfg.get('dim', 128)
        n_layers = cfg.get('n_layers', 2); n_heads = cfg.get('n_heads', 2); dropout = cfg.get('dropout', 0.2)
        self.item_emb = nn.Embedding(n_items, d, padding_idx=0)
        self.pos_emb  = nn.Embedding(self.max_len, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=4*d, dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, n_items, bias=False)
        self.out.weight = self.item_emb.weight
    def forward(self, x, attn_mask):
        B, L = x.size(); pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.item_emb(x) + self.pos_emb(pos)
        causal_block = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        key_padding = (x == 0)
        h = self.encoder(h, mask=causal_block, src_key_padding_mask=key_padding); h = self.norm(h)
        last_idx = attn_mask.sum(dim=1) - 1; last_idx = last_idx.clamp(min=0)
        hT = h[torch.arange(B, device=x.device), last_idx]; logits = self.out(hT); return logits
    @torch.no_grad()
    def full_scores(self, x, attn_mask): return self.forward(x, attn_mask)
