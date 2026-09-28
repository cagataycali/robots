"""Train, calibrate and evaluate the S1V decider on cached DINOv2 features.

    python -m strands_robots.policies.s1v.train ROOT [ROOT2 ...] --out CKPT_DIR [--no-proprio]

Every episode shard is joined with its feature shard; the split is by
episode. After training, per-head temperatures are fitted on the validation
split (temperature scaling for choice heads, one scalar per noul head) and
stored in the config so ``S1VDecider.probabilities`` is calibrated by
default. ``metrics.json`` holds accuracy, ECE and Brier per head with and
without calibration.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import read_manifest
from .model import CANDIDATE_NONE, CHOICE_HEADS, NOUL_HEADS, S1VConfig, S1VDecider, count_parameters

HOLD_CLASS = CHOICE_HEADS["joint"] - 1


def load_split(
    roots: list[Path], *, val_fraction: float = 0.1, seed: int = 0
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Concatenate every tick of every shard that has features; split by episode."""
    rng = np.random.default_rng(seed)
    parts: list[dict[str, np.ndarray]] = []
    info = {"episodes": 0, "ticks": 0, "roots": [str(r) for r in roots], "skipped_no_features": 0}
    for root in roots:
        root = Path(root)
        for m in read_manifest(root):
            fpath = root / "features" / f"{m['name']}.npz"
            epath = root / "episodes" / f"{m['name']}.npz"
            if not fpath.exists():
                info["skipped_no_features"] += 1
                continue
            with np.load(epath) as e, np.load(fpath) as f:
                n = len(e["state"])
                parts.append(
                    {
                        "cams": np.concatenate([f["scene"], f["wrist"]], axis=1),  # (T, 34, 384) f16
                        "state": e["state"],
                        "task": np.full(n, int(e["task"]), dtype=np.int64),
                        "expert_idx": e["expert_idx"].astype(np.int64),
                        "executed_idx": e["executed_idx"].astype(np.int64),
                        "factors": e["expert_factors"].astype(np.int64),
                        "labels": e["labels"],
                        "phase": e["phase"].astype(np.int64),
                        "episode": np.full(n, info["episodes"], dtype=np.int64),
                    }
                )
                info["episodes"] += 1
                info["ticks"] += n
    if not parts:
        raise RuntimeError("load_split: no featurized episodes found")
    ep_ids = np.arange(info["episodes"])
    rng.shuffle(ep_ids)
    n_val = max(1, int(round(val_fraction * len(ep_ids))))
    val_set = set(ep_ids[:n_val].tolist())
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    is_val = np.isin(cat["episode"], list(val_set))
    train = {k: v[~is_val] for k, v in cat.items()}
    val = {k: v[is_val] for k, v in cat.items()}
    info["train_ticks"] = int(len(train["state"]))
    info["val_ticks"] = int(len(val["state"]))
    info["val_episodes"] = n_val
    return train, val, info


def to_device(d: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    """Move every array of a split onto ``device`` as tensors."""
    out = {}
    for k, v in d.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.to(device)
    return out


def losses(model: S1VDecider, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    """Two passes share the batch: an unconditioned one for the choice heads and a conditioned one for the nouls."""
    cams, state, task = batch["cams"], batch["state"], batch["task"]
    none = torch.full_like(task, CANDIDATE_NONE)
    out_choice = model(cams, state, task, none)
    out_noul = model(cams, state, task, batch["executed_idx"])
    f = batch["factors"]
    is_move = f[:, 0] != HOLD_CLASS
    l_joint = F.cross_entropy(out_choice["joint"], f[:, 0])
    if is_move.any():
        l_dir = F.cross_entropy(out_choice["direction"][is_move], f[is_move, 1])
        l_size = F.cross_entropy(out_choice["size"][is_move], f[is_move, 2])
    else:
        l_dir = l_size = out_choice["joint"].sum() * 0.0
    l_noul = sum(
        F.binary_cross_entropy_with_logits(out_noul[q], batch["labels"][:, i]) for i, q in enumerate(NOUL_HEADS)
    )
    total = l_joint + l_dir + l_size + l_noul
    parts = {"joint": l_joint.item(), "direction": l_dir.item(), "size": l_size.item(), "noul": l_noul.item()}
    return total, parts


@torch.inference_mode()
def predict(model: S1VDecider, data: dict[str, torch.Tensor], batch_size: int = 1024) -> dict[str, torch.Tensor]:
    """Raw logits for every tick: choice heads unconditioned, noul heads conditioned on the executed primitive."""
    model.eval()
    outs: dict[str, list[torch.Tensor]] = {q: [] for q in (*CHOICE_HEADS, *NOUL_HEADS)}
    n = len(data["task"])
    for s in range(0, n, batch_size):
        sl = slice(s, s + batch_size)
        none = torch.full_like(data["task"][sl], CANDIDATE_NONE)
        oc = model(data["cams"][sl], data["state"][sl], data["task"][sl], none)
        on = model(data["cams"][sl], data["state"][sl], data["task"][sl], data["executed_idx"][sl])
        for q in CHOICE_HEADS:
            outs[q].append(oc[q])
        for q in NOUL_HEADS:
            outs[q].append(on[q])
    return {q: torch.cat(v) for q, v in outs.items()}


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def fit_temperature(logits: torch.Tensor, target: torch.Tensor, *, noul: bool) -> float:
    """1-D temperature by LBFGS on NLL (Guo et al. 2017)."""
    logits = logits.detach().clone()
    log_t = torch.zeros(1, device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        z = logits / log_t.exp()
        loss = F.binary_cross_entropy_with_logits(z, target) if noul else F.cross_entropy(z, target)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def evaluate(model: S1VDecider, data: dict[str, torch.Tensor], *, calibrated: bool) -> dict[str, Any]:
    """Accuracy / ECE / NLL per choice head, Brier / ECE / AUC per noul head, primitive exact match."""
    logits = predict(model, data)
    probs = model.probabilities(logits, calibrated=calibrated)
    f = data["factors"].cpu().numpy()
    is_move = f[:, 0] != HOLD_CLASS
    out: dict[str, Any] = {}
    pred_factors = []
    for i, q in enumerate(CHOICE_HEADS):
        p = probs[q].cpu().numpy()
        pred = p.argmax(-1)
        pred_factors.append(pred)
        mask = np.ones(len(pred), dtype=bool) if q == "joint" else is_move
        correct = (pred[mask] == f[mask, i]).astype(np.float64)
        conf = p[mask].max(-1)
        t = model.cfg.temperatures.get(q, 1.0) if calibrated else 1.0
        tmask = torch.from_numpy(mask).to(logits[q].device)
        out[q] = {
            "accuracy": float(correct.mean()),
            "ece": ece(conf, correct),
            "nll": float(F.cross_entropy(logits[q][tmask] / t, data["factors"][tmask, i]).item()),
            "n": int(mask.sum()),
        }
    pf = np.stack(pred_factors, axis=1)
    exact = (pf[:, 0] == f[:, 0]) & ((pf[:, 0] == HOLD_CLASS) | ((pf[:, 1] == f[:, 1]) & (pf[:, 2] == f[:, 2])))
    out["primitive_exact"] = float(exact.mean())
    out["hold_fraction_true"] = float((~is_move).mean())
    labels = data["labels"].cpu().numpy()
    for i, q in enumerate(NOUL_HEADS):
        p = probs[q].cpu().numpy()
        y = labels[:, i]
        pred = (p > 0.5).astype(np.float64)
        out[q] = {
            "accuracy": float((pred == y).mean()),
            "brier": float(((p - y) ** 2).mean()),
            "ece": ece(np.where(p > 0.5, p, 1 - p), (pred == y).astype(np.float64)),
            "positive_rate": float(y.mean()),
            "auc": auc(p, y),
        }
    return out


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """Rank-based ROC AUC (NaN when a class is absent)."""
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def train(
    roots: list[Path],
    out: Path,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    use_proprio: bool = True,
    seed: int = 0,
    device: str = "cuda",
    d_model: int = 256,
    n_layers: int = 3,
) -> dict[str, Any]:
    """Train on ``roots``, pick the best validation epoch, fit temperatures, write the checkpoint + ``metrics.json``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va, info = load_split(roots, seed=seed)
    print(f"[train] {info}", flush=True)
    tr_t, va_t = to_device(tr, device), to_device(va, device)
    cfg = S1VConfig(use_proprio=use_proprio, d_model=d_model, n_layers=n_layers)
    model = S1VDecider(cfg).to(device)
    print(f"[train] parameters={count_parameters(model)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    n = len(tr_t["task"])
    steps_per_epoch = math.ceil(n / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.1)
    history = []
    best = (float("inf"), None)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        agg: dict[str, float] = {}
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            batch = {k: v[idx] for k, v in tr_t.items()}
            loss, parts = losses(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v / steps_per_epoch
        model.eval()
        with torch.inference_mode():
            vloss = 0.0
            nb = 0
            for s in range(0, len(va_t["task"]), 1024):
                batch = {k: v[s : s + 1024] for k, v in va_t.items()}
                loss_v, _ = losses(model, batch)
                vloss += loss_v.item()
                nb += 1
            vloss /= max(nb, 1)
        ev = evaluate(model, va_t, calibrated=False)
        row = {
            "epoch": epoch,
            "train": agg,
            "val_loss": vloss,
            "val_joint_acc": ev["joint"]["accuracy"],
            "val_exact": ev["primitive_exact"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"[train] ep{epoch:02d} loss={sum(agg.values()):.3f} val={vloss:.3f} joint_acc={ev['joint']['accuracy']:.3f} exact={ev['primitive_exact']:.3f} {row['seconds']:.0f}s",
            flush=True,
        )
        if vloss < best[0]:
            best = (vloss, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    # calibration on the validation split
    logits = predict(model, va_t)
    temps = {}
    for i, q in enumerate(CHOICE_HEADS):
        mask = (
            torch.ones(len(va_t["task"]), dtype=torch.bool, device=device)
            if q == "joint"
            else va_t["factors"][:, 0] != HOLD_CLASS
        )
        temps[q] = fit_temperature(logits[q][mask], va_t["factors"][mask, i], noul=False)
    for i, q in enumerate(NOUL_HEADS):
        temps[q] = fit_temperature(logits[q], va_t["labels"][:, i], noul=True)
    model.cfg.temperatures = temps
    metrics = {
        "info": info,
        "parameters": count_parameters(model),
        "best_val_loss": best[0],
        "temperatures": temps,
        "uncalibrated": evaluate(model, va_t, calibrated=False),
        "calibrated": evaluate(model, va_t, calibrated=True),
        "history": history,
        "args": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "use_proprio": use_proprio,
            "seed": seed,
        },
    }
    out = Path(out)
    model.save_pretrained(out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-proprio", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    a = ap.parse_args(argv)
    m = train(
        a.roots,
        a.out,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        use_proprio=not a.no_proprio,
        seed=a.seed,
        device=a.device,
        d_model=a.d_model,
        n_layers=a.layers,
    )
    print(json.dumps({k: m[k] for k in ("parameters", "best_val_loss", "temperatures")}, default=float))
    print(json.dumps(m["calibrated"], indent=1, default=float))


if __name__ == "__main__":
    main()
