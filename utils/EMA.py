"""
Exponential Moving Average (EMA) Helper.

Maintains a shadow copy of model parameters updated via exponential decay:
    shadow = mu * shadow + (1 - mu) * param

Useful for stabilizing training and improving final model quality.
"""

import torch
import torch.nn as nn


class EMAHelper:
    """Tracks and applies exponential moving average of model parameters.

    Args:
        mu: EMA decay rate (closer to 1 = slower update, default 0.999).
    """

    def __init__(self, mu: float = 0.999):
        self.mu = mu
        self.shadow: dict = {}

    def register(self, module: nn.Module) -> None:
        """Initialize shadow parameters from a module's current state."""
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module: nn.Module) -> None:
        """Update shadow parameters: shadow = mu * shadow + (1 - mu) * param."""
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    (1.0 - self.mu) * param.data + self.mu * self.shadow[name].data
                )

    def ema(self, module: nn.Module) -> None:
        """Copy shadow parameters back to the module (in-place)."""
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module: nn.Module) -> nn.Module:
        """Create a new module instance with EMA parameters loaded.

        Note: This method relies on the module having a ``config`` attribute
        and a ``config.device`` field. For generic use, consider deep-copy
        as an alternative.
        """
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(
                inner_module.config.device
            )
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())

        self.ema(module_copy)
        return module_copy

    def state_dict(self) -> dict:
        """Return shadow parameters for checkpointing."""
        return self.shadow

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore shadow parameters from a checkpoint."""
        self.shadow = state_dict
