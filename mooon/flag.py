from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

classes = __all__ = ["FLAG"]


class FLAG(nn.Module):
    r"""The Free Large-scale Adversarial Augmentation (FLAG)
    from the `"Robust Optimization as Data Augmentation
    for Large-scale Graphs" <https://arxiv.org/abs/2010.09891>`_ paper

    Parameters
    ----------
    criterion : Callable
        The loss function to be used for training the model.
    steps : int, optional
        The number of steps to be taken for adversarial trainin,
        by default 3
    step_size : float, optional
        The size of the perturbation to be added to the input at each step,
        by default 1e-3

    Example
    -------
    .. code-block:: python

        import torch
        from mooon import FLAG
        data = ... # PyG-like data
        model = ... # GNN model
        optimizer = torch.optim.Adam()
        criterion = torch.nn.CrossEntropycriterion()
        flag = FLAG(criterion)

        def forward(perturb):
            out = model(data.x + perturb, data.edge_index, data.edge_attr)
            return out[data.train_mask]

        def train():
            model.train()
            optimizer.zero_grad()
            loss = flag(forward, data.x, data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            return float(loss)

    train()

    Reference:

    https://github.com/devnkong/FLAG
    """
    def __init__(
        self,
        criterion: Callable,
        steps: int = 3,
        step_size: float = 1e-3,
    ):
        super().__init__()

        self.criterion = criterion
        self.steps = steps
        self.step_size = step_size

    def forward(self, forward: Callable, x: Tensor, y: Tensor) -> Tensor:
        r"""Performs forward pass and adversarial training with FLAG algorithm.

        Parameters
        ----------
        forward : Callable
            The self-defined forward function of the model,
            which accepts obj:`perturb` as input.
        x : Tensor
            The input node features.
        y : Tensor
            The target node labels.

        Returns
        -------
        Tensor
            The loss after adversarial training.
        """
        criterion = self.criterion
        step_size = self.step_size

        perturb = torch.empty_like(x).uniform_(-step_size, step_size)
        perturb.requires_grad_()
        out = forward(perturb)
        loss = criterion(out, y) / self.steps

        for _ in range(self.steps - 1):
            loss.backward()
            perturb_data = perturb.detach() + step_size * torch.sign(
                perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0

            out = forward(perturb)
            loss = criterion(out, y) / self.steps

        return loss

    def __repr__(self):
        return (f"{self.__class__.__name__}(criterion={self.criterion}, "
                f"steps={self.steps}, step_size={self.step_size})")
