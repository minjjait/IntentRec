import argparse, json
from pathlib import Path
from .trainer import run_one

# ALL_MODELS = ['gru4rec','sasrec','bert4rec','srgnn']#,'gcegnn','intentgraphrec','pop'
ALL_MODELS = ['pop'] #'gcegnn', 'intentgraphrec',


def main(args):
    results = {}
    for m in ALL_MODELS:
        print(f"=== Running {m} ===")
        r = run_one(dataset=args.dataset, data_path=args.data_path, outdir=args.outdir,
                    model_name=m, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                    dim=args.dim, dropout=args.dropout, n_layers=args.n_layers, n_heads=args.n_heads,
                    max_len=args.max_len, lr=args.lr, wd=args.wd)
        # Compact console summary
        vm, tm = r['valid'], r['test']
        def pick(d, ks): return ', '.join([f"{k}:{d.get(k, float('nan')):.4f}" for k in ks])
        ks = ['recall@5','recall@10','recall@20','ndcg@5','ndcg@10','ndcg@20']
        print(f"---> {m} | VALID {{" + pick(vm, ks) + "}} | TEST {" + pick(tm, ks) + "}")
        results[m] = r
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.outdir)/'summary.json','w',encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('Saved summary to', Path(args.outdir)/'summary.json')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='movielens | lastfm (or pass --data_path)')
    ap.add_argument('--data_path', default=None)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--n_layers', type=int, default=2)
    ap.add_argument('--n_heads', type=int, default=2)
    ap.add_argument('--max_len', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=20250908)
    args = ap.parse_args(); main(args)
