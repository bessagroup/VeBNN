# ------------------ Beginning of Reference Python Module ---------------------
""" This module contains the implementation of the variance network,  which is
an important component of Variance estimation Bayesian Neural Networks (VeBNN).


VarianceNet: class.
    This module expects a backbone network `net` that learns the variance
    function. As for the output, it forwards the output of the backbone network
    and the first output is alpha and the second output is beta. They should be
    positive. In addition, it also output the log prior of the network
    parameters based on the provided prior mean and prior standard deviation.

"""
#
#                                                                       Modules
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================




class GammaVarNet(nn.Module):
    """
    Generic Gamma variance wrapper.

    This module expects a backbone network `net` that outputs 2 * output_size
    features along the last dimension. It:
      - optionally calls the backbone in Bayesian mode (bayes_train=True)
      - applies softplus to enforce positivity
      - splits the last dimension into (alpha, beta)

    It works for both 2D outputs (MLP: [B, 2*D]) and 3D outputs
    (GRU: [B, T, 2*D]), and should also work for higher ranks as long as
    the last dimension is 2*D.
    """

    def __init__(self,
                 net: nn.Module,
                 prior_mu: float = 0.0,
                 prior_sigma: float = 1.0) -> None:
        super().__init__()
        
        self.net = net
        # prior parameters for the neural network parameters
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward pass of the variance network."""


        out = self.net(x)

        # split the output into alpha and beta
        alpha, beta = torch.chunk(out, 2, dim=-1)

        # apply softplus to ensure positivity
        alpha = F.softplus(alpha)
        beta = F.softplus(beta) 

        return alpha, beta
        

    def neg_log_prior(self) -> torch.Tensor:
        """
        Compute the log prior of all parameters of the backbone network
        under an independent Gaussian prior N(prior_mu, prior_sigma^2).

        Returns
        -------
        Tensor
            Scalar tensor containing the log prior.
        """
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
    

# ------------------ End of Reference Python Module ------------------------- #