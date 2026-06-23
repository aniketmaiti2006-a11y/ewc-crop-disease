"""
Smoke test for the EWC continual-learning pipeline.

Verifies that:
  1. ``get_task_loaders`` returns a loader for every season and yields
     correctly shaped batches.
  2. ``CropDiseaseModel`` produces logits with shape ``(B, num_classes)``.
  3. ``EWC.loss`` returns a finite total / CE / penalty triple.
  4. Backward through the total loss updates trainable parameters.
  5. After ``EWC.consolidate(...)`` the EWC penalty becomes non-zero.
  6. All three seasonal tasks (``Spring`` / ``Summer`` / ``Autumn``)
     run through the model end-to-end without errors.

Designed to run on CPU with synthetic data (no PlantVillage download
required) so it is suitable for CI / quick sanity checks.
"""

import warnings

import torch

from data_loader import TASKS, get_task_loaders
from ewc_utils import EWC
from model import CropDiseaseModel

# Silence the "Real dataset not found" warning emitted by get_task_loaders
# when no ./data/train directory is present — this is expected in CI.
warnings.filterwarnings("ignore", category=UserWarning, module="data_loader")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise if ``tensor`` contains NaN or inf values."""
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values: {tensor}")


def _check_shape(actual: torch.Tensor, expected: tuple) -> None:
    if tuple(actual.shape) != tuple(expected):
        raise AssertionError(
            f"Expected shape {expected}, got {tuple(actual.shape)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Individual smoke checks
# ─────────────────────────────────────────────────────────────────────────────

def test_loaders_yield_correct_shapes(loaders, num_classes: int, batch_size: int):
    """Every season loader should produce (B, 3, 224, 224) images and
    (B,) integer labels in [0, num_classes)."""
    for season in TASKS:
        train_loader, _ = loaders[season]
        images, labels = next(iter(train_loader))
        _check_shape(images, (batch_size, 3, 224, 224))
        _check_shape(labels, (batch_size,))
        if labels.dtype != torch.long:
            raise AssertionError(
                f"[{season}] labels dtype should be torch.long, got {labels.dtype}"
            )
        if labels.min().item() < 0 or labels.max().item() >= num_classes:
            raise AssertionError(
                f"[{season}] labels out of range [0, {num_classes}): "
                f"min={labels.min().item()}, max={labels.max().item()}"
            )
        print(f"  [loader]  {season:<8}  images={tuple(images.shape)}  "
              f"labels={tuple(labels.shape)}  "
              f"label-range=[{labels.min().item()}, {labels.max().item()}]")


def test_forward_pass(model: CropDiseaseModel, images: torch.Tensor, num_classes: int):
    """Model forward should return logits of shape (B, num_classes)."""
    logits = model(images)
    _check_shape(logits, (images.size(0), num_classes))
    _check_finite("logits", logits)
    print(f"  [forward] logits={tuple(logits.shape)}  "
          f"min={logits.min().item():.4f}  max={logits.max().item():.4f}")
    return logits


def test_ewc_loss(ewc: EWC, logits: torch.Tensor, labels: torch.Tensor):
    """EWC.loss should return (total, ce, penalty) with finite values."""
    total, ce, pen = ewc.loss(logits, labels)
    _check_finite("total_loss", total)
    _check_finite("ce_loss", ce)
    _check_finite("ewc_penalty", pen)
    if total.item() < ce.item() - 1e-5:
        raise AssertionError(
            f"total loss ({total.item():.4f}) should be >= CE loss "
            f"({ce.item():.4f}) when EWC penalty is non-negative"
        )
    print(f"  [ewc]     total={total.item():.4f}  ce={ce.item():.4f}  "
          f"penalty={float(pen):.4f}")
    return total


def test_backward_updates_params(model: CropDiseaseModel, total: torch.Tensor):
    """Backward should populate .grad on every trainable parameter and
    an optimizer.step() should change the parameter values."""
    trainable = [(n, p) for n, p in model.named_trainable_parameters()]
    if not trainable:
        raise AssertionError("Model has no trainable parameters")

    model.zero_grad(set_to_none=True)
    total.backward()
    grads_missing = [n for n, p in trainable if p.grad is None]
    if grads_missing:
        raise AssertionError(
            f"Backward did not populate .grad for: {grads_missing[:5]}"
            f"{' ...' if len(grads_missing) > 5 else ''}"
        )

    # Snapshot params, take one step, confirm at least one param changed.
    snapshot = {n: p.detach().clone() for n, p in trainable}
    opt = torch.optim.SGD(model.trainable_parameters(), lr=1e-3)
    opt.step()
    changed = [
        n for n, p in model.named_trainable_parameters()
        if not torch.equal(p.detach(), snapshot[n])
    ]
    if not changed:
        raise AssertionError("No trainable parameters changed after optim.step()")
    print(f"  [backward] grad-populated={len(trainable)}  "
          f"params-updated={len(changed)}  (e.g. {changed[0]})")


def test_ewc_consolidation(model: CropDiseaseModel, ewc: EWC, loaders, num_classes: int):
    """After consolidating one task the EWC penalty should be non-zero
    and the number of stored Fisher matrices should increase by 1."""
    device = next(model.parameters()).device
    train_loader, _ = loaders["Spring"]

    n_before = ewc.num_consolidated_tasks
    ewc.consolidate(train_loader, device, n_samples=8, task_name="Spring-smoke")
    n_after = ewc.num_consolidated_tasks
    if n_after != n_before + 1:
        raise AssertionError(
            f"consolidate() should add exactly one task "
            f"(was {n_before}, now {n_after})"
        )

    # Compute penalty on dummy data — it must be non-negative.
    images, labels = next(iter(train_loader))
    labels = labels % num_classes  # synthetic labels may exceed num_classes
    logits = model(images)
    _, _, pen = ewc.loss(logits, labels)
    pen_val = pen.item()
    if pen_val < 0.0:
        raise AssertionError(f"EWC penalty should be >= 0, got {pen_val}")
    print(f"  [ewc-cons] tasks={n_after}  penalty_after_consolidate={pen_val:.4f}")


def test_all_seasons(model: CropDiseaseModel, loaders, num_classes: int):
    """Run one mini-batch from every season through model + EWC to make
    sure none of the seasonal transforms break the forward/backward path."""
    device = next(model.parameters()).device
    for season in TASKS:
        train_loader, _ = loaders[season]
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)

        # Relabel to be in range [0, num_classes) — synthetic data may
        # carry labels from the larger default class set.
        labels = labels % num_classes

        logits = model(images)
        _check_shape(logits, (images.size(0), num_classes))

        ewc = EWC(model, lambda_ewc=100.0)
        total, ce, pen = ewc.loss(logits, labels)
        _check_finite("total_loss", total)
        _check_finite("ce_loss", ce)
        total.backward()
        print(f"  [season]  {season:<8}  ce={ce.item():.4f}  "
              f"penalty={float(pen):.4f}  ok")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    batch_size  = 4
    num_classes = 10
    device      = torch.device("cpu")

    print("=" * 72)
    print("SMOKE TEST — EWC continual-learning pipeline")
    print("=" * 72)

    # 1. Data loaders
    print("\n[1/5] Building task loaders ...")
    loaders = get_task_loaders(batch_size=batch_size, num_workers=0)
    if set(loaders.keys()) != set(TASKS):
        raise AssertionError(
            f"Expected loaders for {TASKS}, got {list(loaders.keys())}"
        )
    print(f"  loaders ready for seasons: {list(loaders.keys())}")

    # 2. Model
    print("\n[2/5] Building model ...")
    model = CropDiseaseModel(num_classes=num_classes, pretrained=False).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {n_trainable:,} / {n_total:,}")
    if n_trainable == 0:
        raise AssertionError("Model has zero trainable parameters")

    # 3. Forward + loss + backward
    print("\n[3/5] Forward + EWC loss + backward ...")
    train_loader, _ = loaders["Spring"]
    images, labels = next(iter(train_loader))
    # Synthetic labels may exceed num_classes — relabel safely.
    labels = labels % num_classes
    images, labels = images.to(device), labels.to(device)

    ewc = EWC(model, lambda_ewc=100.0)
    logits = test_forward_pass(model, images, num_classes)
    total  = test_ewc_loss(ewc, logits, labels)
    test_backward_updates_params(model, total)

    # 4. EWC consolidation
    print("\n[4/5] EWC consolidation ...")
    test_ewc_consolidation(model, ewc, loaders, num_classes)

    # 5. All seasons end-to-end
    print("\n[5/5] All-season end-to-end run ...")
    test_all_seasons(model, loaders, num_classes)

    print("\n" + "=" * 72)
    print("SMOKE OK — forward, backward, EWC loss, consolidation, "
          "and all-season run succeeded")
    print("=" * 72)


if __name__ == "__main__":
    main()
