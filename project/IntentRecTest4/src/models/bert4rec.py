# src/models/bert4rec.py
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, cfg, n_items):
        super().__init__()
        self.max_len = int(cfg.get("max_len", 100))
        d        = int(cfg.get("dim", 128))
        n_layers = int(cfg.get("n_layers", 2))
        n_heads  = int(cfg.get("n_heads", 2))
        dropout  = float(cfg.get("dropout", 0.2))

        self.item_emb = nn.Embedding(n_items, d, padding_idx=0)
        self.pos_emb  = nn.Embedding(self.max_len, d)

        enc = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=4 * d,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        # 🔧 핵심: Nested Tensor 최적화 끄기 (빈 prefix 샘플 허용)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers, enable_nested_tensor=False)

        self.norm = nn.LayerNorm(d)
        self.out  = nn.Linear(d, n_items, bias=False)
        # weight tying
        self.out.weight = self.item_emb.weight

    def forward(self, x, attn_mask):
        """
        x: (B, L) int64, PAD=0
        attn_mask: (B, L) {0,1}
        """
        B, L = x.size()
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h   = self.item_emb(x) + self.pos_emb(pos)

        # BERT4Rec: 양방향 self-attention (causal mask 사용하지 않음)
        key_padding = (x == 0)  # bool
        h = self.encoder(h, src_key_padding_mask=key_padding)
        h = self.norm(h)

        # 마지막 유효 토큰 위치의 표현을 사용
        last_idx = attn_mask.sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        hT = h[torch.arange(B, device=x.device), last_idx]
        return self.out(hT)

    @torch.no_grad()
    def full_scores(self, x, attn_mask):
        return self.forward(x, attn_mask)
