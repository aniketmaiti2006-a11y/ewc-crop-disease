import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import CropDiseaseModel
from ewc_utils import EWC
from data_loader import get_task_loaders, TASKS
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sequential EWC training for crop disease classification"
    )
    p.add_argument("--data_root",   type=str,   default="./data")
    p.add_argument("--save_dir",    type=str,   default="./checkpoints")
    p.add_argument("--epochs",      type=int,   default=15)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--lambda_ewc",  type=float, default=500.0)
    p.add_argument("--num_classes", type=int,   default=10)
    p.add_argument("--fisher_samples", type=int, default=512)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: CropDiseaseModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds  = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total if total else 0.0


def compute_bwt(
    R_tt:       Dict[str, float],
    R_Tt:       Dict[str, float],
    task_order: List[str],
) -> float:
    diffs = [R_Tt[t] - R_tt[t] for t in task_order[:-1]]
    return float(np.mean(diffs))


def print_banner(text: str, width: int = 64) -> None:
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def train_one_task(
    task_name:    str,
    model:        CropDiseaseModel,
    ewc:          EWC,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    epochs:       int,
    lr:           float,
    use_ewc:      bool,
) -> Tuple[float, List[Dict]]:
    print_banner(
        f"Task: {task_name}  |  EWC={'ON (lambda=' + str(ewc.lambda_ewc) + ')' if use_ewc else 'OFF'}"
    )

    optimizer = optim.Adam(
        model.trainable_parameters(), lr=lr, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    epoch_log: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        running_total = running_ce = running_ewc_pen = 0.0
        correct = total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            if use_ewc:
                loss, ce_val, ewc_val = ewc.loss(logits, labels)
            else:
                ce_val  = nn.CrossEntropyLoss()(logits, labels)
                ewc_val = torch.tensor(0.0, device=device)
                loss    = ce_val

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_total   += loss.item()
            running_ce      += ce_val.item()
            running_ewc_pen += ewc_val.item()

            preds          = logits.argmax(dim=1)
            correct        += (preds == labels).sum().item()
            total_samples  += labels.size(0)

        scheduler.step()

        train_acc = 100.0 * correct / total_samples
        val_acc   = evaluate(model, val_loader, device)
        elapsed   = time.time() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        n_batches = len(train_loader)
        log_entry = {
            "epoch":     epoch,
            "loss":      running_total   / n_batches,
            "ce":        running_ce      / n_batches,
            "ewc":       running_ewc_pen / n_batches,
            "train_acc": train_acc,
            "val_acc":   val_acc,
        }
        epoch_log.append(log_entry)

        print(
            f"  [{epoch:02d}/{epochs}] "
            f"loss={log_entry['loss']:.4f} "
            f"(ce={log_entry['ce']:.4f}, ewc={log_entry['ewc']:.4f}) | "
            f"train={train_acc:.2f}%  val={val_acc:.2f}%  "
            f"[{elapsed:.1f}s]"
        )

    print(f"  [BEST] Best val accuracy for {task_name}: {best_val_acc:.2f}%")
    return best_val_acc, epoch_log


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    print_banner(
        f"EWC Crop Disease | IEEE Paper | lambda={args.lambda_ewc} | "
        f"epochs={args.epochs} | device={device}"
    )

    print("\n[1/5]  Building seasonal data loaders...")
    loaders = get_task_loaders(
        root              = args.data_root,
        batch_size        = args.batch_size,
        num_workers       = args.num_workers,
        num_classes       = args.num_classes,
    )
    train_loaders = {t: loaders[t][0] for t in TASKS}
    val_loaders   = {t: loaders[t][1] for t in TASKS}

    print("\n[2/5]  Initialising model and EWC...")
    model = CropDiseaseModel(
        num_classes  = args.num_classes,
        pretrained   = not args.no_pretrain,
        freeze_early = False,
    ).to(device)

    ewc = EWC(model, lambda_ewc=args.lambda_ewc)
    print(f"  Model parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  {ewc}")

    print("\n[3/5]  Sequential continual learning...")

    R_tt: Dict[str, float] = {}

    full_log: Dict[str, List] = {}

    for i, task in enumerate(TASKS):
        use_ewc_flag = (i > 0)

        best_acc, ep_log = train_one_task(
            task_name    = task,
            model        = model,
            ewc          = ewc,
            train_loader = train_loaders[task],
            val_loader   = val_loaders[task],
            device       = device,
            epochs       = args.epochs,
            lr           = args.lr,
            use_ewc      = use_ewc_flag,
        )

        R_tt[task]       = best_acc
        full_log[task]   = ep_log

        if task != TASKS[-1]:
            print(f"\n[EWC] Consolidating after {task}...")
            ewc.consolidate(
                data_loader = train_loaders[task],
                device      = device,
                n_samples   = args.fisher_samples,
                task_name   = task,
            )
            chk_path = os.path.join(args.save_dir, f"model_{task}.pt")
            torch.save(model.state_dict(), chk_path)
            print(f"  Checkpoint saved -> {chk_path}")

    print("\n[4/5]  Final evaluation on all seasonal tasks...")
    print_banner("Final Accuracy Matrix")
    R_Tt: Dict[str, float] = {}

    for task in TASKS:
        acc         = evaluate(model, val_loaders[task], device)
        R_Tt[task]  = acc
        delta       = acc - R_tt[task]
        sign        = "+" if delta >= 0 else ""
        print(
            f"  {task:<10} | R_tt={R_tt[task]:6.2f}%  "
            f"R_Tt={acc:6.2f}%  Delta={sign}{delta:.2f}%"
        )

    print("\n[5/5]  Computing Backward Transfer...")
    bwt = compute_bwt(R_tt, R_Tt, list(TASKS))

    print_banner(f"Backward Transfer (BWT) = {bwt:.2f}%")
    threshold = -2.9
    if bwt >= threshold:
        print(f"  [PASS]  BWT {bwt:.2f}% >= {threshold}%  "
              f"-> Catastrophic-forgetting criterion PASSED")
    else:
        print(f"  [FAIL]  BWT {bwt:.2f}% < {threshold}%  "
              f"-> Consider increasing lambda or fisher_samples")

    results = {
        "lambda_ewc": args.lambda_ewc,
        "epochs":     args.epochs,
        "R_tt":       R_tt,
        "R_Tt":       R_Tt,
        "BWT":        bwt,
        "epoch_logs": full_log,
    }
    results_path = os.path.join(args.save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {results_path}")

    final_chk = os.path.join(args.save_dir, "model_final.pt")
    torch.save(model.state_dict(), final_chk)
    print(f"  Final model  -> {final_chk}")


if __name__ == "__main__":
    main()
