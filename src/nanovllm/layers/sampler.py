import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy_mask = temperatures == 0
        greedy_tokens = torch.argmax(logits, dim=-1)

        # For non-greedy, use gumbel-max.
        # To avoid division by zero, we can replace 0s with 1s in temperatures,
        # as they will be masked out later anyway.
        safe_temperatures = torch.where(greedy_mask, torch.ones_like(temperatures), temperatures)
        
        logits = logits.float().div_(safe_temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        
        return torch.where(greedy_mask, greedy_tokens, sample_tokens)
