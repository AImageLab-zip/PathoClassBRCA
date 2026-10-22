from collections import Counter

import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        return output.sum() / mask.sum()


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    return criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])


def compute_class_weights_multitask(task_labels: dict, cfg: dict, device="cuda", normalize=True, ignore_index=-1):
    weights_by_task = {}

    for task_name, labels in task_labels.items():
        if task_name not in cfg["tasks"]:
            raise ValueError(f"Task '{task_name}' not found in cfg['tasks']")

        num_classes = int(cfg["tasks"][task_name]["num_classes"])

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().tolist()
            
        labels = [int(l) for l in labels if l != ignore_index]

        if not labels:
            raise ValueError(f"[{task_name}] no annotated samples in the training set")

        out_of_range = sorted({l for l in labels if not 0 <= l < num_classes})
        if out_of_range:
            raise ValueError(f"[{task_name}] labels out of range [0, {num_classes}): {out_of_range}")

        counts = Counter(labels)
        total = len(labels)
        freqs = [counts.get(c, 0) for c in range(num_classes)]

        missing = [c for c, f in enumerate(freqs) if f == 0]
        if missing:
            print(f"[{task_name}] WARNING: classes absent from the training set (weight = 0): {missing}")

        w = torch.tensor(
            [total / f if f > 0 else 0.0 for f in freqs],
            dtype=torch.float32,
            device=device,
        )

        if normalize:
            n_present = num_classes - len(missing)
            w = w / w.sum() * n_present

        for c in range(num_classes):
            print(f"[{task_name}] class {c}: freq={freqs[c]} weight={w[c].item():.6f}")

        weights_by_task[task_name] = w

    return weights_by_task