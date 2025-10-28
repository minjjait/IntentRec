import torch
def recall_at_k(ranks, K): return (ranks < K).float().mean().item()
def ndcg_at_k(ranks, K):
    gains = 1.0 / torch.log2(ranks.float()+2.0); return (gains * (ranks < K).float()).mean().item()
def mrr(ranks): return (1.0 / (ranks.float()+1.0)).mean().item()
