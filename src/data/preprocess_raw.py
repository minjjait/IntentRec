import argparse, json
from pathlib import Path
import pandas as pd, numpy as np

def iterative_kcore(df, user_col, item_col, min_items_user, min_users_item):
    changed = True
    while changed:
        changed = False
        uc = df[user_col].value_counts()
        keep_u = uc[uc >= min_items_user].index
        if len(keep_u) < df[user_col].nunique():
            df = df[df[user_col].isin(keep_u)]; changed = True
        ic = df[item_col].value_counts()
        keep_i = ic[ic >= min_users_item].index
        if len(keep_i) < df[item_col].nunique():
            df = df[df[item_col].isin(keep_i)]; changed = True
    return df

def preprocess(in_path, out_path, user_col, item_col, time_col, 
               min_items_user=5, min_users_item=5, min_seq_len=3,
               dedup_consecutive=True, time_ascending=True):
    p = Path(in_path); df = pd.read_csv(p)
    cols = [user_col, item_col, time_col]
    for c in cols:
        if c not in df.columns: raise ValueError(f"Column '{c}' not found in {list(df.columns)}")
    df = df[cols].dropna()
    df[user_col] = df[user_col].astype(int)
    df[item_col] = df[item_col].astype(int)
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col]); df[time_col] = df[time_col].astype(np.int64)
    df = df.sort_values([user_col, time_col], ascending=time_ascending, kind='mergesort')
    if dedup_consecutive:
        df['_prev_item'] = df.groupby(user_col)[item_col].shift(1)
        df = df[df[item_col] != df['_prev_item']].drop(columns=['_prev_item'])
    df = iterative_kcore(df, user_col, item_col, min_items_user, min_users_item)
    seqs = (df.groupby(user_col)[item_col].apply(list).reset_index(name='seq'))
    seqs = seqs[seqs['seq'].apply(len) >= min_seq_len]
    all_items = sorted({i for s in seqs['seq'] for i in s})
    item2id = {it:i+1 for i,it in enumerate(all_items)}
    id2item = {v:k for k,v in item2id.items()}
    seqs['seq_mapped'] = seqs['seq'].apply(lambda s: [item2id[i] for i in s])
    proc = pd.DataFrame({
        'user_id': seqs[user_col].astype(int).values,
        'item_sequence': [' '.join(map(str, s)) for s in seqs['seq_mapped'].tolist()]
    })
    out_path = Path(out_path); proc.to_csv(out_path, index=False, encoding='utf-8')
    lens = [len(s) for s in seqs['seq_mapped']]
    rep_rate = 0.0; n_rep = 0
    for s in seqs['seq_mapped']:
        if len(s) >= 2 and s[-1] == s[-2]: n_rep += 1
    if len(lens) > 0: rep_rate = n_rep / len(lens)
    stats = {
        'users': int(proc.shape[0]), 'items': int(len(all_items)),
        'avg_seq_len': float(np.mean(lens)) if lens else 0.0,
        'min_seq_len': int(min(lens) if lens else 0), 'max_seq_len': int(max(lens) if lens else 0),
        'consecutive_repeat_rate': float(rep_rate), 'min_items_user': int(min_items_user),
        'min_users_item': int(min_users_item), 'dedup_consecutive': bool(dedup_consecutive),
    }
    (out_path.with_suffix('.stats.json')).write_text(json.dumps(stats, indent=2), encoding='utf-8')
    (out_path.with_suffix('.mapping.json')).write_text(json.dumps({'item2id': item2id, 'id2item': id2item}, indent=2), encoding='utf-8')
    print('Saved:', str(out_path)); print('Stats:', stats)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True, help='Path to raw interactions CSV')
    ap.add_argument('--out', dest='out_path', required=True, help='Path to output processed CSV')
    ap.add_argument('--user_col', default='userId'); ap.add_argument('--item_col', default='itemId')
    ap.add_argument('--time_col', default='timestamp')
    ap.add_argument('--min_items_user', type=int, default=5)
    ap.add_argument('--min_users_item', type=int, default=5)
    ap.add_argument('--min_seq_len', type=int, default=3)
    ap.add_argument('--no_dedup_consecutive', action='store_true')
    ap.add_argument('--desc_time', action='store_true')
    args = ap.parse_args()
    preprocess(args.in_path, args.out_path, args.user_col, args.item_col, args.time_col,
               args.min_items_user, args.min_users_item, args.min_seq_len,
               dedup_consecutive=not args.no_dedup_consecutive, time_ascending=not args.desc_time)
