# src/eval/evaluator.py
from __future__ import annotations
import torch
import numpy as np
from .metrics import recall_at_k, ndcg_at_k, mrr

class Evaluator:
    """
    Full-vocabulary evaluator (누수 차단 + 안전가드 포함)
    - prefix 마스킹을 모델 입력과 동일한 길이(L=max_len)로 제한하여
      후보 과축소 문제를 방지.
    """
    def __init__(self, n_items: int, device: torch.device | None = None,
                 topk=(5, 10, 20), min_keep_ratio: float = 0.95):
        self.n_items = int(n_items)
        self.device = device or torch.device("cpu")
        self.topk = list(topk)
        self.min_keep_ratio = float(min_keep_ratio)

    def _pad_prefix(self, seq_in: list[int], L: int) -> torch.Tensor:
        x = np.zeros(L, dtype=np.int64)
        if seq_in:
            s = np.asarray(seq_in[-L:], dtype=np.int64)  # 안전하게 마지막 L개만
            x[-len(s):] = s
        return torch.tensor(x, device=self.device).unsqueeze(0)

    def _make_prefix(self, seq: list[int], tgt: int) -> list[int]:
        # 타깃의 마지막 등장 이전까지만 prefix (검증 미래 누수 차단)
        try:
            pos = max(i for i, v in enumerate(seq) if int(v) == int(tgt))
            return seq[:pos]
        except ValueError:
            return seq

    @torch.no_grad()
    def evaluate(self, model, sequences: list[list[int]], targets: list[int]) -> dict:
        model.eval()
        ranks = []
        L = getattr(model, "max_len", 100)

        for seq, tgt in zip(sequences, targets):
            tgt = int(tgt)

            # 1) 타깃 이전 prefix 생성 + 마지막 L개로 컷 (중요!)
            seq_in_full = self._make_prefix(seq, tgt)
            seq_in = seq_in_full[-L:]  # 모델 입력과 동일한 길이만 사용

            # 2) 모델 입력
            x = self._pad_prefix(seq_in, L)
            attn = (x != 0).long()

            # 3) FP32 평가
            with torch.amp.autocast("cuda", enabled=False):
                scores = model.full_scores(x, attn)

            # 4) 점수 sanity check
            assert scores.ndim == 2 and scores.shape[0] == 1, f"bad scores shape {scores.shape}"
            assert scores.shape[1] == self.n_items, f"logits dim {scores.shape[1]} != n_items {self.n_items}"

            scores = torch.nan_to_num(scores, nan=-1e9, neginf=-1e9, posinf=1e9).clamp(-50, 50)

            # 5) PAD + prefix(타깃 제외) 마스킹  — prefix는 마지막 L개만 반영
            scores[0, 0] = -1e9
            forbid = set(map(int, seq_in))
            forbid.discard(tgt)
            if forbid:
                idx = torch.tensor(sorted(forbid), device=self.device)
                scores[0, idx] = -1e9

            # 6) 후보 수 가드 (95% 이상 유지)
            allowed = int((scores[0] > -1e9).sum().item())
            need = int(self.n_items * self.min_keep_ratio)
            assert allowed >= need, (
                f"Too few candidates: {allowed}/{self.n_items} "
                f"(prefix_len={len(seq_in)}, masked={self.n_items - allowed})"
            )

            # 7) 타깃이 마스킹되지 않았는지 확인
            assert scores[0, tgt].item() > -1e9 + 1e-6, "target appears masked"

            # 8) 랭크 계산
            tgt_score = scores[0, tgt]
            rank = (scores[0] >= tgt_score).sum().item() - 1
            if rank < 0:
                rank = 0
            ranks.append(rank)

        ranks = torch.tensor(ranks, device=self.device)
        out = {f"recall@{k}": recall_at_k(ranks, k) for k in self.topk}
        out.update({f"ndcg@{k}": ndcg_at_k(ranks, k) for k in self.topk})
        out["mrr"] = mrr(ranks)
        return out