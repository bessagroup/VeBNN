# ------------------ Beginning of Reference Python Module ---------------------
""" This module contains the implementation of the mean network,  an important
component of Variance estimation Bayesian Neural Networks (VeBNN).


MeanNet: class.
    This module expects a backbone network `net` that learns the mean function.
    As for the output, it forwards the output of the backbone network directly.
    In addition, it also output the log prior of the network parameters based
    on the provided prior mean and prior standard deviation.

"""
#
#                                                                       Modules
# =============================================================================
import torch
import torch.nn as nn

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================



class MeanNet(nn.Module):
    """an abstract class for mean networks, which can be extended to create
    various mean network architectures.
    """

    def __init__(self,
                 net: nn.Module,
                 prior_mu: float,
                 prior_sigma: float) -> None:
        super(MeanNet, self).__init__()
        """
        initialize the mean network.
        Parameters
        ----------
        net : nn.Module
            the backbone network for the mean network.
        prior_mu : torch.Tensor
            the prior mean of the weights.
        prior_sigma : torch.Tensor
            the prior standard deviation of the weights.
        """
        # network architecture
        self.net = net
        # prior parameters for the neural network parameters
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of the mean network.

        Parameters
        ----------
        x : torch.Tensor
            input tensor.

        Returns
        -------
        torch.Tensor
            output of the mean network.
        """
        self.net.forward(x)

        return self.net.forward(x)

    def neg_log_prior(self) -> torch.Tensor:
        """
        Compute the log prior of all parameters of the backbone network.

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the log prior.
        """
        # get device from network parameters
        params = list(self.net.parameters())
        if len(params) == 0:
            return torch.tensor(0.0)

        device = params[0].device
        dist = torch.distributions.Normal(
            loc=torch.tensor(self.prior_mu, device=device),
            scale=torch.tensor(self.prior_sigma, device=device),
        )

        log_prior = torch.tensor(0.0, device=device)
        for param in params:
            log_prior += dist.log_prob(param).sum()

        return -log_prior
