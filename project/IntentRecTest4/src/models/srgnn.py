# src/models/srgnn.py
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    SR-GNN의 미니멀 구현.
    - 배치 내 각 세션을 소규모 그래프로 만들고 한 번의 메시지 패싱(GRUCell) 후
      마지막 아이템 기반 attention readout으로 세션 표현을 만든 뒤
      전체 vocab에 대한 점수(logits)를 반환.
    - AMP(autocast) 환경에서 dtype 불일치를 피하기 위해 index_add_ 버퍼를
      반드시 messages와 동일 dtype으로 생성(zeros_like)한다.
    """
    def __init__(self, cfg: Dict, n_items: int):
        super().__init__()
        d = int(cfg.get("dim", 128))
        self.max_len = int(cfg.get("max_len", 100))

        self.item_emb = nn.Embedding(n_items, d, padding_idx=0)

        # incoming/outgoing 변환
        self.lin_in  = nn.Linear(d, d, bias=False)
        self.lin_out = nn.Linear(d, d, bias=False)

        # gated 업데이트 (메시지 2d → 상태 d)
        self.gru = nn.GRUCell(2 * d, d)

        # readout용 투영
        self.read = nn.Linear(d, d, bias=False)

        # 출력: 전 아이템 점수 (weight tying)
        self.out = nn.Linear(d, n_items, bias=False)
        self.out.weight = self.item_emb.weight  # tie

    # ----- single session graph -----
    def _one_graph_logits(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        seq_ids: (L,) int64, PAD=0 포함 가능
        반환: (n_items,) float{16/32} — 전체 vocab 점수
        """
        device = seq_ids.device
        # PAD 제거
        seq = seq_ids[seq_ids != 0]
        # 완전 빈 세션이면 0점 반환
        if seq.numel() == 0:
            d = self.item_emb.weight.size(1)
            return torch.zeros(self.out.out_features, device=device, dtype=self.item_emb.weight.dtype)

        # 등장 순서를 유지한 unique 노드 (노드 id → [0..M-1] 매핑)
        uniq, inv = torch.unique_consecutive(seq, return_inverse=True)  # uniq: (M,), inv maps each step to node id
        node_emb = self.item_emb(uniq)  # (M, d)

        # 엣지: 연속 클릭 간(src→dst)
        if seq.numel() >= 2:
            src_nodes = inv[:-1]
            dst_nodes = inv[1:]

            # 메시지 준비
            Hin  = self.lin_in(node_emb)   # (M, d)  incoming 메시지 소스
            Hout = self.lin_out(node_emb)  # (M, d)  outgoing 메시지 소스

            # AMP 친화적: messages와 같은 dtype으로 버퍼 준비
            agg_in  = torch.zeros_like(Hin)    # (M, d)
            agg_out = torch.zeros_like(Hout)   # (M, d)

            # dst로 incoming 합산, src로 outgoing 합산
            agg_in.index_add_(0, dst_nodes, Hin[src_nodes])     # in-degree
            agg_out.index_add_(0, src_nodes, Hout[dst_nodes])   # out-degree

            agg = torch.cat([agg_in, agg_out], dim=-1)  # (M, 2d)
            h0 = node_emb
            h = self.gru(agg, h0)                       # (M, d)
        else:
            # 길이 1이면 업데이트 없이 그대로 사용
            h = node_emb

        # 마지막 아이템을 query로 attention readout
        last_node = inv[-1]
        q = self.read(h[last_node])                # (d,)
        att = torch.softmax(h @ q, dim=0)          # (M,)
        s = (att.unsqueeze(0) @ h).squeeze(0)      # (d,)

        # 전체 아이템 점수 (weight tying, dtype 정합 보장)
        W = self.item_emb.weight
        if W.dtype != s.dtype:
            W = W.to(s.dtype)
        scores = torch.matmul(s, W.T)              # (n_items,)
        return scores

    # ----- batched -----
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) int64, PAD=0
        attn_mask: (B, L) (사용 안 하지만 인터페이스 일치용)
        반환: (B, n_items) logits
        """
        B, L = x.size()
        out = []
        for i in range(B):
            out.append(self._one_graph_logits(x[i]))
        return torch.stack(out, dim=0)

    @torch.no_grad()
    def full_scores(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self.forward(x, attn_mask)
