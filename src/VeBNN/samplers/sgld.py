# ------------------ Beginning of Reference Python Module ---------------------
""" Module for Stochastic Gradient Langevin Dynamics Sampler using PyTorch.

Classes
-------
SGLD
    Stochastic Gradient Langevin dynamics Sampler that
    uses a burn-in procedure to adapt its own hyperparameters during the
    initial stages of sampling.
"""
#
#                                                                       Modules
# =============================================================================
from torch.autograd import Variable
from torch.optim import Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================


class SGLD(Optimizer):
    """Langevin Stochastic Gradient Descent optimizer. It is used for
    bayesian neural networks. It is a variant of SGD optimizer.

    References
    ----------
    Welling, M., & Teh, Y. W. (2011). "Bayesian learning via stochastic
    gradient Langevin dynamics". In Proceedings of the 28th International
    Conference on International Conference on Machine Learning (pp. 681-688).
    """

    def __init__(self,
                 params: dict,
                 lr: float,
                 nesterov: bool = False) -> None:
        """Initialization of Langevin SGD

        Parameters
        ----------
        params : dict
            A dictionary containing the parameters to optimize.
        lr : float
            Learning rate. Must be positive.
        nesterov : bool, optional
            Whether to use Nesterov momentum, by default False.

        Raises
        ------
        ValueError
            If the learning rate is non-positive.
        """

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        self.netstrov = nesterov
        super(SGLD, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(SGLD, self).__setstate__(state)
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
        float or None
            The loss value if a closure is provided, otherwise None.
        """

        loss = None
        # first case
        if closure is not None:
            loss = closure()
        # loop over the parameters
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # if len(p.shape) == 1 and p.shape[0] == 1:
                #     # for aleatoric noise, no Langevin dynamics is involved.
                #     p.data.add_(d_p, alpha=-group['lr'])
                # else:
                # add unit noise to the weights and bias
                unit_noise = Variable(p.data.new(p.size()).normal_())
                # Langevin dynamics update
                p.data.add_(0.5*d_p, alpha=-group['lr'])
                p.data.add_(unit_noise, alpha=group['lr']**0.5)

        return loss
