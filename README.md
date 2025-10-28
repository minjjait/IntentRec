# recsys-exp-v2.5
Sequential recommendation experiments with **6 models** (POP, GRU4Rec, SASRec, BERT4Rec, SR-GNN, GCE-GNN, IntentGraphRec)
and a **one-shot runner**. Includes a robust **raw→processed** preprocessor and **LOOCV** evaluation (valid predicts t-1; test predicts t).

## Quickstart (Windows PowerShell)
# 0) venv + install deps
pip install -r requirements.txt
# 0-1) PyTorch (CUDA 12.1 for RTX 3060)
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 1) If you already have processed CSV (columns: user_id, item_sequence):
python -m src.run_all --dataset movielens --data_path "C:\data\movielens_processed.csv" --epochs 20 --outdir results\movielens\all

# 2) If you have *raw* interactions (user, item, timestamp):
python -m src.data.preprocess_raw --in "C:\data\ratings.csv" --out "C:\data\movielens_processed.csv" ^
  --user_col userId --item_col movieId --time_col timestamp --min_items_user 5 --min_users_item 5 --min_seq_len 3

#    Then run all models:
sh run.sh

Outputs:
- Per-model metrics: <outdir>/<model>/metrics.json
- Summary of all models: <outdir>/summary.json
Console prints include Recall/NDCG @5/10/20.
