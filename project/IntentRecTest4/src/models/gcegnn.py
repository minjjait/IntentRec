import torch, torch.nn as nn

class Model(nn.Module):
    def __init__(self, cfg, n_items, cooc=None):
        super().__init__()
        self.max_len = cfg.get("max_len", 100); d = cfg.get("dim", 128)
        self.item_emb = nn.Embedding(n_items, d, padding_idx=0)
        self.W_in = nn.Linear(d, d, bias=False); self.W_out = nn.Linear(d, d, bias=False)
        self.gate = nn.GRUCell(d, d)
        self.readout = nn.Linear(3*d, d); self.out = nn.Linear(d, n_items, bias=False)
        self.out.weight = self.item_emb.weight
        self.cooc = cooc
        self.k_neighbors = cfg.get("k_neighbors", 64)
        self.alpha = cfg.get("alpha", 0.3)

    def _edges_from_seq(self, seq):
        nonzero = seq[seq != 0]
        if nonzero.numel() < 2: return None
        nodes, inv = torch.unique(nonzero, return_inverse=True)
        src, dst = inv[:-1], inv[1:]
        return nodes, src, dst, nonzero[-1]

    def _global_vector(self, last_item):
        if self.cooc is None: return None
        key = int(last_item.item())
        if key not in self.cooc: return None
        nbs = list(self.cooc[key].keys())[:self.k_neighbors]
        if not nbs: return None
        nbs = torch.as_tensor(nbs, device=self.item_emb.weight.device, dtype=torch.long)
        return self.item_emb(nbs).mean(dim=0, keepdim=True)  # (1, d)

    def forward(self, x, attn_mask):
        B = x.size(0); device = x.device
        logits = torch.full((B, self.item_emb.num_embeddings), -1e9, device=device)
        logits[:, 0] = 0

        for b in range(B):
            ed = self._edges_from_seq(x[b])
            if ed is None: 
                continue
            nodes, src, dst, last_tok = ed
            H0  = self.item_emb(nodes)
            Hin = self.W_in(H0); Hout = self.W_out(H0)

            N, d = H0.shape
            agg = torch.zeros((N, d), device=device, dtype=H0.dtype)
            agg.index_add_(0, dst, Hout[src])
            deg = torch.bincount(dst, minlength=N).clamp_min(1).unsqueeze(1).to(agg.dtype)
            H = self.gate(Hin + agg/deg, H0)

            idx = (nodes == last_tok).nonzero(as_tuple=True)[0]
            h_last = H[idx] if idx.numel() > 0 else H[-1:]
            att = torch.softmax(H @ h_last.T, dim=0)
            local = (att * H).sum(dim=0, keepdim=True)
            gvec = self._global_vector(last_tok)
            if gvec is None: gvec = torch.zeros_like(local)
            h_read = torch.tanh(self.readout(torch.cat([h_last, local, self.alpha * gvec], dim=-1).squeeze(0)))
            logits[b] = h_read @ self.item_emb.weight.T
        return logits

    @torch.no_grad()
    def full_scores(self, x, attn_mask):
        return self.forward(x, attn_mask)
