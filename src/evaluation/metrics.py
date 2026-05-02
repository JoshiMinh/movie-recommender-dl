from __future__ import annotations

import math
from typing import Dict, Iterable, List

import torch


@torch.no_grad()
def topk_metrics(logits: torch.Tensor, targets: torch.Tensor, ks: Iterable[int]) -> Dict[str, float]:
    ks = sorted(set(int(k) for k in ks))
    max_k = max(ks)
    _, topk_idx = torch.topk(logits, k=max_k, dim=1)

    metrics: Dict[str, float] = {}
    for k in ks:
        preds = topk_idx[:, :k]
        hits = (preds == targets.unsqueeze(1)).any(dim=1).float()

        # Single-target implicit feedback: hit and recall are equivalent.
        hit = hits.mean().item()
        recall = hit

        ndcg_sum = 0.0
        for i in range(preds.size(0)):
            row = preds[i].tolist()
            tgt = int(targets[i].item())
            if tgt in row:
                rank = row.index(tgt) + 1
                ndcg_sum += 1.0 / math.log2(rank + 1)
        ndcg = ndcg_sum / preds.size(0)

        metrics[f"Hit@{k}"] = hit
        metrics[f"Recall@{k}"] = recall
        metrics[f"NDCG@{k}"] = ndcg

    return metrics


@torch.no_grad()
def evaluate_model(model, loader, ks: List[int], device: torch.device) -> Dict[str, float]:
    model.eval()
    aggregate = {f"Hit@{k}": 0.0 for k in ks}
    aggregate.update({f"Recall@{k}": 0.0 for k in ks})
    aggregate.update({f"NDCG@{k}": 0.0 for k in ks})

    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        batch_metrics = topk_metrics(logits, y, ks)

        batch_size = x.size(0)
        count += batch_size
        for key, val in batch_metrics.items():
            aggregate[key] += val * batch_size

    if count == 0:
        return {k: 0.0 for k in aggregate}

    return {k: v / count for k, v in aggregate.items()}


def format_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    if not results:
        return "No results to display."

    metric_keys = sorted(next(iter(results.values())).keys())
    header = ["Model"] + metric_keys

    widths = [len(h) for h in header]
    rows = []
    for model_name, metrics in results.items():
        row = [model_name] + [f"{metrics[k]:.4f}" for k in metric_keys]
        rows.append(row)
        for i, col in enumerate(row):
            widths[i] = max(widths[i], len(col))

    def fmt_row(cols):
        return " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols))

    line = "-+-".join("-" * w for w in widths)
    text_lines = [fmt_row(header), line]
    text_lines.extend(fmt_row(r) for r in rows)
    return "\n".join(text_lines)
