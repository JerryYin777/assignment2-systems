"""Training utilities for CSE599-O Assignment 1."""

import math
import os
from collections.abc import Iterable
from typing import IO, BinaryIO, Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, ""]:
    """Compute cross-entropy loss with numerical stability.

    Formula: ℓ = -log(softmax(logits)[target])

    With numerical stability tricks:
    - Subtract max for numerical stability
    - Cancel log and exp where possible

    Args:
        logits: Unnormalized logits of shape (..., vocab_size)
        targets: Target indices of shape (...) with values in [0, vocab_size-1]

    Returns:
        Average cross-entropy loss across all examples (scalar)
    """
    # Flatten batch dimensions for easier processing
    batch_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]

    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # For numerical stability, subtract the max value from logits
    max_logits = torch.max(logits_flat, dim=-1, keepdim=True).values
    logits_shifted = logits_flat - max_logits

    # Compute log(sum(exp(logits))) for each example
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1))

    # Get the logit values for the target classes
    target_logits = logits_shifted[torch.arange(logits_flat.shape[0]), targets_flat]

    # Cross entropy: -log(softmax(logit)[target])
    # = -(logit[target] - log(sum(exp(logits))))
    # = log(sum(exp(logits))) - logit[target]
    losses = log_sum_exp - target_logits

    # Return average loss across all examples
    return losses.mean()


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer with weight decay.

    Implements Algorithm 1 from Loshchilov & Hutter (2019).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """Initialize AdamW optimizer.

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (α)
            betas: Coefficients (β₁, β₂) for computing running averages
            eps: Term added to denominator for numerical stability (ε)
            weight_decay: Weight decay coefficient (λ)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates model and returns loss

        Returns:
            Loss if closure is provided, else None
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Get or initialize state for this parameter
                state = self.state[p]

                if len(state) == 0:
                    # First time step for this parameter
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)  # First moment
                    state['v'] = torch.zeros_like(p.data)  # Second moment

                # Increment time step (t starts at 1, not 0)
                state['t'] += 1
                t = state['t']

                m = state['m']
                v = state['v']

                # Update biased first moment estimate: m ← β₁m + (1-β₁)g
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment estimate: v ← β₂v + (1-β₂)g²
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected learning rate: αₜ = α * sqrt(1-β₂ᵗ) / (1-β₁ᵗ)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters: θ ← θ - αₜ * m / (sqrt(v) + ε)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                # Apply weight decay: θ ← θ - αλθ
                p.data.mul_(1 - lr * weight_decay)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine annealing learning rate schedule with linear warmup.

    Args:
        it: Current iteration number
        max_learning_rate: α_max, maximum learning rate
        min_learning_rate: α_min, minimum/final learning rate
        warmup_iters: T_w, number of warmup iterations
        cosine_cycle_iters: T_c, number of cosine annealing iterations

    Returns:
        Learning rate at iteration it
    """
    # Warm-up phase: linearly increase from 0 to max_learning_rate
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    # Cosine annealing phase
    elif it <= cosine_cycle_iters:
        # Compute progress through cosine cycle
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # Cosine annealing formula
        return min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (
            max_learning_rate - min_learning_rate
        )

    # Post-annealing phase: constant at min_learning_rate
    else:
        return min_learning_rate


def clip_grad_norm(
    parameters: Iterable[nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
) -> None:
    """Clip gradients by global L2 norm.

    If ||g||₂ > max_norm, scale all gradients by max_norm / (||g||₂ + ε)

    Args:
        parameters: Iterable of parameters whose gradients to clip
        max_norm: Maximum L2 norm
        eps: Small epsilon for numerical stability (default: 1e-6)
    """
    # Collect all parameters with gradients
    params_with_grad = [p for p in parameters if p.grad is not None]

    if len(params_with_grad) == 0:
        return

    # Compute total L2 norm of all gradients
    total_norm = torch.sqrt(
        sum(torch.sum(p.grad.data ** 2) for p in params_with_grad)
    )

    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + eps)

    # Only clip if total_norm > max_norm (clip_coef < 1)
    if clip_coef < 1:
        for p in params_with_grad:
            p.grad.data.mul_(clip_coef)


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample training batches for language modeling.

    For each example in the batch, randomly sample a starting position and
    extract context_length tokens as input and the next token as target.

    This function is memory-efficient and works with both regular numpy arrays
    and memory-mapped arrays (np.memmap). When using memory-mapped arrays,
    only the required slices are loaded into memory, making it suitable for
    large datasets that don't fit in RAM.

    Args:
        dataset: 1D numpy array or np.memmap of token IDs
        batch_size: Number of examples to sample
        context_length: Length of context window
        device: Device to place tensors on ('cpu' or 'cuda:0')

    Returns:
        Tuple of (inputs, targets) where:
        - inputs: shape (batch_size, context_length)
        - targets: shape (batch_size, context_length)
    """
    # We need context_length tokens for input and 1 more for target
    # So valid starting indices are [0, len(dataset) - context_length - 1]
    max_start_idx = len(dataset) - context_length - 1

    # Randomly sample starting indices
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

    # Extract sequences
    # Note: For np.memmap, this only loads the specific slices into memory
    # rather than the entire dataset, making it memory-efficient
    inputs = np.stack([dataset[i:i + context_length] for i in start_indices])
    targets = np.stack([dataset[i + 1:i + context_length + 1] for i in start_indices])

    # Convert to PyTorch tensors
    inputs_tensor = torch.from_numpy(inputs).long().to(device)
    targets_tensor = torch.from_numpy(targets).long().to(device)

    return inputs_tensor, targets_tensor


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Save model, optimizer, and iteration to checkpoint.

    Args:
        model: Model to serialize
        optimizer: Optimizer to serialize
        iteration: Current training iteration number
        out: Path or file-like object to save to
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load model, optimizer, and iteration from checkpoint.

    Args:
        src: Path or file-like object to load from
        model: Model to restore state to
        optimizer: Optimizer to restore state to

    Returns:
        Iteration number from the checkpoint
    """
    checkpoint = torch.load(src, weights_only=False)

    # Handle compiled models (torch.compile adds _orig_mod. prefix)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']
