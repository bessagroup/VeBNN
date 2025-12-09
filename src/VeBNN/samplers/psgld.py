# ------------------ Beginning of Reference Python Module ---------------------
""" Module for preconditioned Stochastic Gradient Langevin Dynamics Sampler
using PyTorch.

Classes
-------
pSGLD
    preconditioned Stochastic Gradient Hamiltonian Monte-Carlo Sampler that
    uses a burn-in procedure to adapt its own hyperparameters during the
    initial stages of sampling.
"""
#
#                                                                       Modules
# =============================================================================
# import Variable
import torch
from torch.autograd import Variable
from torch.optim import Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================


class pSGLD(Optimizer):
    """Preconditioned Stochastic Gradient Langevin Dynamics optimizer.

    This optimizer/sampler is commonly used for Bayesian neural networks. It is
    a variant of Stochastic Gradient Descent (SGD) and Stochastic Gradient
    Langevin Dynamics (SGLD).

    References
    ----------
    Li, C., Chen, C., Carlson, D., & Carin, L. (2016, February). "
    Preconditioned stochastic gradient Langevin dynamics for deep neural
    networks". In Proceedings of the AAAI conference on artificial intelligence
    (Vol. 30, No. 1).
    """

    def __init__(self,
                 params: dict,
                 lr: float,
                 alpha: float = 0.99,
                 eps: float = 1e-5,
                 nesterov: bool = False) -> None:
        """Initialize the pSGLD optimizer.

        Parameters
        ----------
        params : dict
            A dictionary containing the parameters to optimize.
        lr : float
            Learning rate. Must be positive.
        alpha : float, optional
            Coefficient for computing running averages of gradient squares,
            by default 0.99.
        eps : float, optional
            A small constant added for numerical stability, by default 1e-5.
        nesterov : bool, optional
            Whether to use Nesterov momentum, by default False.

        Raises
        ------
        ValueError
            If the learning rate is non-positive.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        self.netstrov = nesterov
        super(pSGLD, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(pSGLD, self).__setstate__(state)
        """Set the state of the optimizer.

        Parameters
        ----------
        state : dict
            The state dictionary containing optimizer state information.
        """
        # change default state values for param groups
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None) -> float:
        """Perform a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that re-evaluates the model and returns the loss,
            by default None.

        Returns
        -------
        float
            The loss value if a closure is provided, otherwise None.
        """

        loss = None
        if closure is not None:
            loss = closure()
        # loop over the parameters
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # get the state
                state = self.state[p]

                # initialize the state
                if len(state) == 0:
                    state['step'] = 0
                    state['V'] = torch.zeros_like(p.data)

                # get the state parameters
                V = state['V']
                # get the parameters
                alpha = group['alpha']
                eps = group['eps']
                lr = group['lr']

                # update the state
                state['step'] += 1

                # update parameters
                # update V
                V.mul_(alpha).addcmul_(1 - alpha, d_p, d_p)

                # get 1/G use the inplace operation (need division when use it)
                G = V.add(eps).sqrt_()

                # update parameters with 0.5*lr (main update of pSGLD)
                p.data.addcdiv_(d_p, G, value=-lr/2)

                # inject noise to the parameters
                noise_std = torch.sqrt(lr/G)
                noise = Variable(p.data.new(p.size()).normal_())
                p.data.add_(noise_std*noise)

        return loss
