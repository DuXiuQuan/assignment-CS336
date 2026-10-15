import torch
from collections.abc import Iterable

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True).values

    x = x - max_val

    exp_x = torch.exp(x)

    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def cross_entropy(
        logits: torch.Tensor,
        targets: torch.Tensor
) -> torch.Tensor:
    max_logits = torch.max(logits, dim = -1, keepdim = True).values

    shift_logits = logits - max_logits

    exp_logits = torch.exp(shift_logits)
    sum_exp = torch.sum(exp_logits, dim = -1)
    log_sum_exp = torch.log(sum_exp)

    correct_logits = shift_logits[
        torch.arange(logits.shape[0], device=logits.device),
        targets
    ]

    loss = log_sum_exp - correct_logits

    return loss.mean()

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    params = [p for p in parameters if p.grad is not None]

    if not params:
        return

    with torch.no_grad():
        total = sum(torch.sum(p.grad ** 2) for p in params)

        grad_norm = torch.sqrt(total)

        if(grad_norm > max_l2_norm):
            scale = max_l2_norm / (grad_norm + 1e-6)
            for p in params:
                p.grad.mul_(scale)