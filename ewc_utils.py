import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from copy import deepcopy


def compute_fisher_matrix(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    n_samples: int = 256,
) -> Dict[str, torch.Tensor]:
    model.eval()

    fisher: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    log_softmax = nn.LogSoftmax(dim=1)
    total_samples = 0

    for images, _labels in data_loader:
        if total_samples >= n_samples:
            break

        images = images.to(device)
        model.zero_grad(set_to_none=True)

        logits   = model(images)
        log_prob = log_softmax(logits)

        probs           = log_prob.exp().detach()
        sampled_classes = torch.multinomial(probs, num_samples=1).squeeze(1)

        nll = -log_prob[range(len(sampled_classes)), sampled_classes].mean()
        nll.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.detach().pow(2)

        total_samples += images.size(0)

    n_batches = max(1, total_samples // data_loader.batch_size)
    for name in fisher:
        fisher[name] /= n_batches

    return fisher


class EWC:

    def __init__(self, model: nn.Module, lambda_ewc: float = 400.0):
        self.model      = model
        self.lambda_ewc = lambda_ewc
        self._ce_loss   = nn.CrossEntropyLoss()

        self._fishers:     List[Dict[str, torch.Tensor]] = []
        self._theta_stars: List[Dict[str, torch.Tensor]] = []

    def consolidate(
        self,
        data_loader: DataLoader,
        device: torch.device,
        n_samples: int = 256,
        task_name: str = "task",
    ) -> None:
        print(f"  [EWC] Consolidating after {task_name} "
              f"(n_samples={n_samples})...", end=" ", flush=True)

        fisher = compute_fisher_matrix(
            self.model, data_loader, device, n_samples
        )

        theta_star: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self._fishers.append(fisher)
        self._theta_stars.append(theta_star)
        print("done.")

    def penalty(self, device: torch.device = None) -> torch.Tensor:
        _device = device or next(self.model.parameters()).device
        ewc_sum = torch.tensor(0.0, device=_device)

        for fisher, theta_star in zip(self._fishers, self._theta_stars):
            for name, param in self.model.named_parameters():
                if not param.requires_grad or name not in fisher:
                    continue
                F_diag    = fisher[name].to(_device)
                theta_old = theta_star[name].to(_device)
                ewc_sum = ewc_sum + (F_diag * (param - theta_old).pow(2)).sum()

        return (self.lambda_ewc / 2.0) * ewc_sum

    def loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce      = self._ce_loss(logits, labels)
        ewc_pen = self.penalty()
        total   = ce + ewc_pen
        return total, ce, ewc_pen

    @property
    def num_consolidated_tasks(self) -> int:
        return len(self._fishers)

    def fisher_norm(self, task_idx: int = -1) -> float:
        f = self._fishers[task_idx]
        total = sum(v.pow(2).sum().item() for v in f.values())
        return total ** 0.5

    def __repr__(self) -> str:
        return (
            f"EWC(lambda={self.lambda_ewc}, "
            f"consolidated_tasks={self.num_consolidated_tasks})"
        )
