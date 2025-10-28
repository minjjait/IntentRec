from src.data.loader import load_processed_csv
from src.trainer import build_model
from src.utils import get_device
import torch, numpy as np, random, statistics as S

dataset = "gowalla"
path = r".\gowalla_processed.csv"

tx,ty,vx,vy,qx,qy,N,pop,cooc,stats = load_processed_csv(dataset, path)

cfg = {"dim":128, "n_layers":2, "n_heads":2, "dropout":0.2, "max_len":100}
m = build_model("bert4rec", cfg, N, pop, cooc).to(get_device()); m.eval()

def inspect(seq, tgt, L=100):
    tgt = int(tgt)
    seq_in = seq[:-1] if (seq and int(seq[-1]) == tgt) else seq
    x = np.zeros(L, np.int64)
    s = np.array(seq_in[-L:], np.int64); x[-len(s):] = s
    x = torch.tensor(x, device=get_device()).unsqueeze(0)
    attn = (x != 0).long()
    # 평가는 FP32 (AMP 끔)
    with torch.amp.autocast("cuda", enabled=False):
        scores = m.full_scores(x, attn)
    scores = torch.nan_to_num(scores, nan=-1e9, neginf=-1e9, posinf=1e9).clamp(-50, 50)
    # prefix 마스킹(타깃 제외는 evaluator에서 처리됨; 여기선 타깃도 제외하지 않음)
    forbid = set(map(int, seq_in))
    if forbid:
        idx = torch.tensor(sorted(forbid), device=get_device())
        scores[0,0] = -1e9; scores[0,idx] = -1e9
    allowed = int((scores[0] > -1e9).sum().item())
    # 동점 이득 없음
    rank = int((scores[0] >= scores[0, tgt]).sum().item()) - 1
    if rank < 0: rank = 0
    return allowed, rank

random.seed(0)
idxs = random.sample(range(len(qx)), 200)
allowed, ranks = [], []
for i in idxs:
    a, r = inspect(qx[i], qy[i])
    allowed.append(a); ranks.append(r)

print("n_items:", N)
print("allowed min/mean/max:", min(allowed), int(S.mean(allowed)), max(allowed))
print("approx R@20:", sum(1 for r in ranks if r < 20)/len(ranks))
print("rank stats min/median/max:", min(ranks), sorted(ranks)[len(ranks)//2], max(ranks))
