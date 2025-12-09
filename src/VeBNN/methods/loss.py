# ------------------ Beginning of Reference Python Module ---------------------
""" Module for Negative Log-Likelihood for Gaussian distribution, referred as
NLL Loss. It is used for evaluating the discrepancy between the predicted
distribution and the real distribution.

Classes
-------
NLLLoss
    Negative Log-Likelihood Loss for Gaussian Distribution.
GammaNLLLoss
    Negative Log-Likelihood Loss for Gamma Distribution.
"""
#
#                                                                       Modules
# =============================================================================
import torch
from torch.nn import Module

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================


class NLLLoss(Module):
    """Calculate the negative log likelihood loss
    """

    def __init__(self, reduction: str = "sum") -> None:
        """initialization
        """
        super(NLLLoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                pred_var: torch.Tensor,
                real: torch.Tensor,
                num_scale: int = 1) -> torch.Tensor:
        """calculate the negative log likelihood loss

        Parameters
        ----------
        pred : torch.Tensor
            prediction of BNN
        real : torch.Tensor
            responses of training data
        pred_var : torch.Tensor
            predicted variance (learned noise)
        num_scale : int
            number of batches (default is 1)

        Returns
        -------
        torch.Tensor
            negative log likelihood function calculation
        """
        # adding a small value to avoid log(0)
        # var = var + 1e-6

        exponent = -0.5*(pred - real)**2/pred_var
        log_coef = -0.5*torch.log(pred_var)

        neg_lld = - (log_coef + exponent) - 0.5 * \
            torch.log(torch.tensor(2*torch.pi))

        if self.reduction == "sum":
            neg_lld = neg_lld.sum()
        elif self.reduction == "mean":
            neg_lld = neg_lld.mean()
        else:
            raise ValueError("Undefined reduction type")
        
        # divide by the number of outputs
        neg_lld = neg_lld*num_scale

        return neg_lld


class GammaNLLLoss(Module):
    """Negative Log-Likelihood Loss for Gamma Distribution"""

    def __init__(self, reduction: str = 'mean'):
        """Initialize the Gamma Negative Log-Likelihood Loss.

        Parameters:
        reduction: str, optional,
            Specifies the reduction to apply to output, by default 'mean'.
            Possible values are 'mean' and 'sum'.
        """

        super(GammaNLLLoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                residuals: torch.Tensor,
                alpha: torch.Tensor,
                beta: torch.Tensor,
                num_scale: int = 1):
        """Compute the Negative Log-Likelihood Loss for Gamma Distribution.

        Parameters:
        -----------
        residuals: torch.Tensor,
            The residuals of the regression model.
        alpha: torch.Tensor,
            The shape parameter of the Gamma distribution.
        beta: torch.Tensor,
            The rate parameter of the Gamma distribution.
        num_scale: int, optional,
            The number of scales in the Gamma distribution, by default 1.

        Returns:
        --------
        loss: torch.Tensor,
            the Negative Log-Likelihood Loss for Gamma Distribution.

        Raises:
        -------
        ValueError
            If the reduction is not 'mean' or 'sum'.
        """

        # add a small number to avoid numerical instability
        eps = 1e-12
        residuals = residuals + eps
        # calculate the loss
        loss = (
            alpha * torch.log(beta + eps)
            - torch.lgamma(alpha)
            + (alpha - 1) * torch.log(residuals + eps)
            - beta * residuals
        )
        # apply reduction
        if self.reduction == 'mean':
            return -torch.mean(loss)
        elif self.reduction == 'sum':
            return -torch.sum(loss)*num_scale
        else:
            raise ValueError("Invalid reduction: {}".format(self.reduction))
