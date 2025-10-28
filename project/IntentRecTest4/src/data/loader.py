from pathlib import Path
import pandas as pd, numpy as np
from torch.utils.data import Dataset
from collections import Counter, defaultdict

class SeqDataset(Dataset):
    def __init__(self, sequences, targets, max_len=100):
        self.sequences, self.targets, self.max_len = sequences, targets, max_len
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        import torch
        seq, tgt = self.sequences[idx], self.targets[idx]
        x = np.zeros(self.max_len, dtype=np.int64)
        seq = np.array(seq[-self.max_len:], dtype=np.int64)
        x[-len(seq):] = seq
        attn_mask = (x != 0).astype(np.int64)
        return torch.from_numpy(x), torch.tensor(tgt, dtype=torch.long), torch.from_numpy(attn_mask)

def load_processed_csv(name, data_path=None):
    if data_path: fname = Path(data_path)
    else:
        fname = {"movielens":"movielens_processed.csv","lastfm":"lastfm_processed.csv"}.get(name)
        if fname: fname = Path(fname)
        else: raise ValueError("Unknown dataset; pass --data_path to CSV")
    if not Path(fname).exists():
        alt = Path('/mnt/data')/Path(fname).name
        if alt.exists(): fname = alt
    df = pd.read_csv(fname)
    seqs = []
    for s in df['item_sequence'].astype(str).tolist():
        toks = [int(t) for t in s.strip().split() if t.strip().isdigit()]
        seqs.append(toks)
    # Map items to 1..N (0=PAD)
    all_items = sorted(set(i for seq in seqs for i in seq))
    item2id = {it:i+1 for i,it in enumerate(all_items)}
    remapped = [[item2id[i] for i in seq] for seq in seqs]
    # Popularity & global co-occurrence (window=3)
    pop = Counter(i for seq in remapped for i in seq)
    cooc = defaultdict(Counter)
    for seq in remapped:
        L = len(seq)
        for t in range(L):
            for j in range(max(0,t-3), min(L, t+4)):
                if j==t: continue
                a, b = seq[t], seq[j]
                cooc[a][b] += 1
    n_items = len(item2id)+1
    # LOOCV splits + sliding window training
    train_inputs, train_targets, valid_seq, valid_tgt, test_seq, test_tgt = [], [], [], [], [], []
    for seq in remapped:
        if len(seq) < 3: 
            continue
        for i in range(1, len(seq)-1):  # predict seq[i] from prefix up to i-1
            train_inputs.append(seq[:i]); train_targets.append(seq[i])
        v, t = seq[-2], seq[-1]
        valid_seq.append(seq[:-2]); valid_tgt.append(v)
        test_seq.append(seq[:-1]);  test_tgt.append(t)
    stats = {
        "n_items": n_items, "n_train": len(train_targets), "n_valid": len(valid_tgt),
        "n_test": len(test_tgt), "avg_seq_len": float(np.mean([len(s) for s in remapped])) if remapped else 0.0,
    }
    return (train_inputs, train_targets, valid_seq, valid_tgt, test_seq, test_tgt, n_items, pop, cooc, stats)

def collate_fn(batch):
    xs, ys, masks = zip(*batch); import torch
    return torch.stack(xs), torch.stack(ys), torch.stack(masks)
