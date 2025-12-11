# ------------------ Beginning of Reference Python Module ---------------------
""" Module for preconditioned Stochastic Gradient Langevin Dynamics Sampler
using PyTorch.

Classes
-------
SGHMC
    Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
    procedure to adapt its own hyperparameters during the initial stages
    of sampling.
"""
#
#                                                                       Modules
# =============================================================================
import torch
from torch.optim import Optimizer
import numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================


class SGHMC(Optimizer):
    """Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
    procedure to adapt its own hyperparameters during the initial stages
    of sampling.

    Note: this code is took from: https://github.com/automl/pybnn

    References
    ----------
    [1] T. Chen, E. B. Fox, C. Guestrin, In Proceedings of Machine Learning
    Research 32 (2014). _`Stochastic Gradient Hamiltonian Monte Carlo
    <https://arxiv.org/pdf/1402.4102.pdf>`_

    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        mdecay: float = 0.01,
        wd: float = 0.00002,
        scale_grad: float = 1.0,
    ) -> None:
        """Set up a SGHMC Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr: float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        mdecay:float, optional
            (Constant) momentum decay per time-step.
            Default: `0.05`.
        scale_grad: float, optional
            Value that is used to scale the magnitude of the noise used
            during sampling. In a typical batches-of-data setting this usually
            corresponds to the number of examples in the entire dataset.
            Default: `1.0`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, scale_grad=scale_grad, mdecay=mdecay, wd=wd)
        super().__init__(params, defaults)

    def step(self, closure: any = None):
        """Perform a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that re-evaluates the model and returns the loss,
            by default None.

        Returns
        -------
        float or None
            The loss value if a closure is provided, otherwise None.
        """
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.randn(
                        parameter.size(), dtype=parameter.dtype
                    )

                state["iteration"] += 1

                mdecay, lr, _ = group["mdecay"], group["lr"], group["wd"]
                scale_grad = group["scale_grad"]

                momentum = state["momentum"]
                gradient = parameter.grad.data * scale_grad
                # set beta to be zero in this case
                sigma = torch.sqrt(
                    torch.from_numpy(
                        np.array(2 * lr * mdecay, dtype=type(lr)))
                )
                sample_t = torch.normal(
                    mean=torch.zeros_like(gradient),
                    std=torch.ones_like(gradient) * sigma,
                )

                # update the momentum and parameters according to algorithm
                parameter.data.add_(lr * mdecay * momentum)
                momentum.add_(-lr * gradient - mdecay *
                              lr * momentum + sample_t)
        return loss
