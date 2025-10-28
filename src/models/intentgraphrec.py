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
        self.norm = nn.LayerNorm(d)
        self.W_in = nn.Linear(d, d, bias=False); self.W_out = nn.Linear(d, d, bias=False)
        self.gate_cell = nn.GRUCell(d, d)
        self.fuse = nn.Linear(2*d, d); self.out = nn.Linear(d, n_items, bias=False)
        self.out.weight = self.item_emb.weight
    def _local_graph(self, seq):
        nodes = torch.unique(seq[seq!=0])
        if nodes.numel()==0: return nodes, torch.zeros(0,0,device=seq.device)
        idx_map = {int(n.item()):i for i,n in enumerate(nodes)}
        edges = []; nonzero = seq[seq!=0].tolist()
        for a,b in zip(nonzero[:-1], nonzero[1:]): edges.append((idx_map[a], idx_map[b]))
        N = nodes.numel(); A = torch.zeros(N,N, device=seq.device)
        for i,j in edges: A[i,j] += 1.0
        d = A.sum(dim=1, keepdim=True)+1e-8; A = A/d
        return nodes, A
    def forward(self, x, attn_mask):
        B, L = x.size(); device = x.device
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        emb = self.item_emb(x); h_tr = emb + self.pos_emb(pos)
        causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        key_padding = (x == 0)
        h_tr = self.encoder(h_tr, mask=causal, src_key_padding_mask=key_padding); h_tr = self.norm(h_tr)
        last_idx = attn_mask.sum(dim=1) - 1; last_idx = last_idx.clamp(min=0)
        hT_tr = h_tr[torch.arange(B, device=x.device), last_idx]
        h_read_list = []
        for b in range(B):
            seq = x[b]; nodes, A = self._local_graph(seq)
            if nodes.numel()==0:
                h_read_list.append(torch.zeros_like(hT_tr[b:b+1])); continue
            H0 = self.item_emb(nodes); Hin = self.W_in(H0); Hout = self.W_out(H0)
            H = self.gate_cell(Hin + A @ Hout, H0)
            last = int(seq[(seq!=0)].tolist()[-1])
            last_idx_node = (nodes==last).nonzero(as_tuple=True)[0]
            h_last = H[last_idx_node] if last_idx_node.numel()>0 else H[-1:]
            att = torch.softmax(H @ h_last.T, dim=0); local = (att*H).sum(dim=0, keepdim=True)
            h_read_list.append(local)
        h_local = torch.cat(h_read_list, dim=0)
        fused = torch.tanh(self.fuse(torch.cat([hT_tr, h_local], dim=-1)))
        logits = fused @ self.item_emb.weight.T
        return logits
    @torch.no_grad()
    def full_scores(self, x, attn_mask): return self.forward(x, attn_mask)
