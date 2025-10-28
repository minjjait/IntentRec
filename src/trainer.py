# src/trainer.py
from __future__ import annotations
import os
import json
import time
import importlib
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from .utils import set_seed, get_device
from .eval.evaluator import Evaluator
from .data.loader import load_processed_csv

cudnn.benchmark = True


# ----------------------------- Dataset ---------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, seqs: List[List[int]], tgts: List[int], max_len: int):
        self.seqs = seqs
        self.tgts = tgts
        self.L = int(max_len)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]
        y = int(self.tgts[i])
        # left-pad
        x = torch.zeros(self.L, dtype=torch.long)
        if s:
            s = s[-self.L:]
            x[-len(s):] = torch.tensor(s, dtype=torch.long)
        attn = (x != 0).long()
        return x, attn, torch.tensor(y, dtype=torch.long)


def _make_loaders(train_ds: Dataset, valid_ds: Dataset, test_ds: Dataset, batch_size: int):
    
    loader_args_train = dict(
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    loader_args_eval = dict(
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
        persistent_workers=True,
    )
    return (
        DataLoader(train_ds, **loader_args_train),
        DataLoader(valid_ds, **loader_args_eval),
        DataLoader(test_ds, **loader_args_eval),
    )


# ----------------------------- Models ----------------------------------------
class PopModel(nn.Module):
    
    def __init__(self, n_items: int, counts: torch.Tensor | None = None, max_len: int = 100):
        super().__init__()
        self.n_items = int(n_items)
        self.max_len = int(max_len)
        if counts is None:
            scores = torch.zeros(self.n_items)
        else:
            scores = counts.float().clamp(min=0)
        self.register_buffer("scores", scores)

    def forward(self, x, attn):
        return self.full_scores(x, attn)

    @torch.no_grad()
    def full_scores(self, x, attn):
        B = x.size(0)
        return self.scores.unsqueeze(0).expand(B, -1)


def build_model(model_name: str, cfg: dict, n_items: int,
                train_item_counts: torch.Tensor | None = None):
    
    name = model_name.lower()
    if name == "pop":
        m = PopModel(n_items=n_items, counts=train_item_counts, max_len=cfg.get("max_len", 100))
        return m

    pkg = __package__.split('.')[0] if __package__ else "src"
    try:
        mod = importlib.import_module(f"{pkg}.models.{name}")
        ModelCls = getattr(mod, "Model")
    except Exception as e:
        raise ImportError(f"cannot import model '{name}' from {pkg}.models.{name} : {e}")

    m = ModelCls(cfg, n_items)
    return m


# ----------------------------- Train/Eval ------------------------------------
@dataclass
class TrainConfig:
    dataset: str
    data_path: str
    outdir: str
    model_name: str
    epochs: int = 20
    batch_size: int = 256
    seed: int = 42
    # common model hparams
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 2
    max_len: int = 100
    dropout: float = 0.2
    # optim
    lr: float = 5e-4
    optim: str = "adamw"
    weight_decay: float = 0.0


def _count_items(seqs: List[List[int]], n_items: int) -> torch.Tensor:
    
    c = torch.zeros(n_items, dtype=torch.float32)
    for s in seqs:
        for v in s:
            iv = int(v)
            if 0 <= iv < n_items:
                c[iv] += 1.0
    return c


def _fmt_metrics(m: dict) -> str:
    
    return "{" + ", ".join(f"{k}:{m[k]:.4f}" for k in sorted(m)) + "}"


def _print_epoch_progress(model_name: str, ep: int, epochs: int, loss: float, t0: float):
    
    import time as _time
    pct = ep / max(1, epochs)
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "·" * (bar_len - filled)
    elapsed = _time.time() - t0
    eta = (elapsed / ep) * (epochs - ep) if ep > 0 else 0.0
    eta_m, eta_s = int(eta // 60), int(eta % 60)
    print(
        f"\r[{model_name}] epochs {ep:>2}/{epochs:<2} {bar} {pct*100:5.1f}% | "
        f"loss={loss:6.4f} | ETA {eta_m}m{eta_s:02d}s",
        end="",
        flush=True,
    )


def train_one_epoch(model, loader, device, criterion, optimizer, scaler,
                    verbose: bool = False, log_interval: int = 200, grad_clip: float = 1.0):
    
    model.train()
    total, steps = 0.0, 0
    for step, (x, attn, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        attn = attn.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x, attn)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        # clip은 unscale 후에
        scaler.unscale_(optimizer)
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total += float(loss.item())
        steps += 1

        if verbose and (step % log_interval == 0):
            if device.type == "cuda":
                a = torch.cuda.memory_allocated() / 1024 ** 2
                r = torch.cuda.memory_reserved() / 1024 ** 2
                print(f"[gpu] step={step:5d} alloc={a:.0f}MB reserved={r:.0f}MB loss={loss.item():.4f}")
            else:
                print(f"[cpu] step={step:5d} loss={loss.item():.4f}")
    return total / max(1, steps)


def evaluate_all(model, evaluator: Evaluator,
                 valid: Tuple[List[List[int]], List[int]],
                 test: Tuple[List[List[int]], List[int]]):
    vx, vy = valid
    qx, qy = test
    valid_metrics = evaluator.evaluate(model, vx, vy)
    test_metrics = evaluator.evaluate(model, qx, qy)
    return valid_metrics, test_metrics


# ----------------------------- Public API ------------------------------------
def run_one(dataset: str,
            data_path: str,
            outdir: str,
            model_name: str,
            epochs: int = 20,
            batch_size: int = 256,
            seed: int = 42,
            verbose: bool = False,             
            show_progress: bool = True,        
            **cfg) -> dict:
    
    os.makedirs(outdir, exist_ok=True)

    # --- seed & device
    set_seed(seed)
    device = get_device()
    if verbose:
        print(f"[run_one] dataset={dataset} model={model_name} outdir={outdir}")
        print(">>> device:", device)

    # --- load data
    tx, ty, vx, vy, qx, qy, N, *_ = load_processed_csv(dataset, data_path)
    n_items = int(N)

    # --- Datasets & Loaders
    max_len = int(cfg.get("max_len", 100))
    train_ds = SequenceDataset(tx, ty, max_len=max_len)
    valid_ds = SequenceDataset(vx, vy, max_len=max_len)
    test_ds = SequenceDataset(qx, qy, max_len=max_len)
    train_loader, valid_loader, test_loader = _make_loaders(train_ds, valid_ds, test_ds, batch_size)

    # --- model
    train_counts = _count_items(tx, n_items) if model_name.lower() == "pop" else None
    model = build_model(
        model_name,
        dict(
            dim=int(cfg.get("dim", 128)),
            n_layers=int(cfg.get("n_layers", 2)),
            n_heads=int(cfg.get("n_heads", 2)),
            max_len=max_len,
            dropout=float(cfg.get("dropout", 0.2)),
        ),
        n_items=n_items,
        train_item_counts=train_counts,
    ).to(device)

    if verbose:
        on = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else device
        print(">>> model on:", on)

    evaluator = Evaluator(n_items=n_items, device=device, topk=(5, 10, 20))

    if model_name.lower() == "pop":
        valid_metrics, test_metrics = evaluate_all(model, evaluator, (vx, vy), (qx, qy))
        print(f"[{model_name}] VALID metrics:")
        for k, v in valid_metrics.items():
            print(f"  {k}: {v:.6f}")
        print(f"[{model_name}] TEST  metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}")
        print(f"---> {model_name} | VALID {_fmt_metrics(valid_metrics)} | TEST {_fmt_metrics(test_metrics)}")
        result = {"model": model_name, "valid": valid_metrics, "test": test_metrics}
        with open(os.path.join(outdir, f"{model_name}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    # --- criterion/optimizer
    criterion = nn.CrossEntropyLoss()
    default_lr = 5e-4 if model_name.lower() != "gru4rec" else 3e-4
    lr = float(cfg.get("lr", default_lr))
    wd = float(cfg.get("weight_decay", 0.0))
    opt_name = str(cfg.get("optim", "adamw")).lower()
    if opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    last_valid, last_test, last_loss = None, None, None
    t0 = time.time()
    for ep in range(1, int(epochs) + 1):
        ep_loss = train_one_epoch(
            model, train_loader, device, criterion, optimizer, scaler,
            verbose=verbose, log_interval=200, grad_clip=1.0
        )

        valid_metrics, test_metrics = evaluate_all(model, evaluator, (vx, vy), (qx, qy))

        state = {"model": model_name, "epoch": ep, "valid": valid_metrics, "test": test_metrics, "loss": ep_loss}
        with open(os.path.join(outdir, f"{model_name}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        if show_progress and ep < int(epochs):
            _print_epoch_progress(model_name, ep, int(epochs), ep_loss, t0)

        if ep == int(epochs):
            if show_progress:
                print("\r" + " " * 120, end="\r")
            last_valid, last_test, last_loss = valid_metrics, test_metrics, ep_loss
            print(f"[{model_name}] epoch {ep} | loss {ep_loss:.4f}")
            print(f"[{model_name}] VALID metrics:")
            for k, v in last_valid.items():
                print(f"  {k}: {v:.6f}")
            print(f"[{model_name}] TEST  metrics:")
            for k, v in last_test.items():
                print(f"  {k}: {v:.6f}")
            print(f"---> {model_name} | VALID {_fmt_metrics(last_valid)} | TEST {_fmt_metrics(last_test)}")

    elapsed = round(time.time() - t0, 2)
    final = {
        "model": model_name,
        "epoch": int(epochs),
        "valid": last_valid,
        "test": last_test,
        "loss": last_loss,
        "_elapsed_sec": elapsed,
    }
    with open(os.path.join(outdir, f"{model_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    return final
