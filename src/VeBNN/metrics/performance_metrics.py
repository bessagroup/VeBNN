# ------------------ Beginning of Reference Python Module ---------------------
""" This module contains the performance metrics for evaluating the model.

Functions
---------
eval_mse: evaluate the mean square error
eval_tll: evaluate the total log likelihood at the original scale
eval_wasserstein_value: evaluate the wasserstein value

"""
#
#                                                                       Modules
# =============================================================================

import numpy as np
import torch
from statistics import NormalDist


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================


def eval_mse(y_true: torch.Tensor | np.ndarray,
             y_pred: torch.Tensor | np.ndarray,
             reduction: str = "mean") -> float:
    """evaluate the error for rnn model

    Parameters
    ----------
    y_true : torch.Tensor
        real output
    y_pred : torch.Tensor
        predicted output

    Returns
    -------
    float
        mean square error
    """

    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)

    if reduction == "mean":
        mse = torch.mean((y_true - y_pred).norm(dim=1) /
                         y_true.norm(dim=1)) * 100
        return mse.item()
    elif reduction == "all":
        mse = (y_true - y_pred).norm(dim=1) / y_true.norm(dim=1) * 100
        return torch.mean(mse, axis=1)


def eval_tll(y_true: torch.Tensor | np.ndarray,
             y_pred: torch.Tensor | np.ndarray,
             y_var: torch.Tensor | np.ndarray,
             reduction: str = "mean") -> float:
    """evaluate the total log likelihood at the original scale

    Parameters
    ----------
    y_true : torch.Tensor
        real output
    y_pred : torch.Tensor
        predicted output
    y_var : float
        total uncertainty (predictive variance)

    Returns
    -------
    float
        total log likelihood for test data
    """

    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    if isinstance(y_var, np.ndarray):
        y_var = torch.tensor(y_var)

    exponent = -0.5 * (y_true - y_pred) ** 2 / y_var
    log_coef = -0.5 * torch.log(y_var)
    tll = log_coef + exponent - 0.5 * torch.log(torch.tensor(2*torch.pi))

    if reduction == "mean":
        tll = torch.mean(tll).item()
        return tll
    elif reduction == "all":
        return tll.mean(axis=[1, 2])


def eval_wasserstein_value(y_true: torch.Tensor,
                           y_true_std: torch.Tensor,
                           y_pred: torch.Tensor,
                           y_pred_std: torch.Tensor,
                           reduction: str = "mean",
                           normalize: bool = False) -> float:
    """Evaluate the Wasserstein-like distance between prediction and ground truth.

    Parameters
    ----------
    y_true : torch.Tensor or np.ndarray
        Ground truth output.
    y_true_std : torch.Tensor or np.ndarray
        Ground truth aleatoric uncertainty.
    y_pred : torch.Tensor or np.ndarray
        Predicted mean.
    y_pred_std : torch.Tensor or np.ndarray
        Predicted aleatoric uncertainty.
    reduction : str, optional
        'mean' for scalar output, 'none' for raw tensor output.
    normalize : bool, optional
        Whether to use normalized Wasserstein.

    Returns
    -------
    float or torch.Tensor
        Wasserstein distance.
    """

    # Ensure torch tensors
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_true_std, np.ndarray):
        y_true_std = torch.tensor(y_true_std, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if isinstance(y_pred_std, np.ndarray):
        y_pred_std = torch.tensor(y_pred_std, dtype=torch.float32)

    eps = 1e-6

    if not normalize:
        wasserstein = torch.sqrt((y_true - y_pred) ** 2 +
                                 (y_true_std - y_pred_std) ** 2)
    else:
        wasserstein = torch.sqrt(
            ((y_true - y_pred) / (torch.abs(y_true) + eps)) ** 2 +
            ((y_true_std - y_pred_std) / (y_true_std + eps)) ** 2
        )

    if reduction == "mean":
        return wasserstein.mean().item()
    elif reduction == "none":
        return wasserstein
    else:
        raise ValueError("reduction must be 'mean' or 'none'")

def eval_picp(y_true: torch.Tensor | np.ndarray,
              y_lower: torch.Tensor | np.ndarray,
              y_upper: torch.Tensor | np.ndarray,
              reduction: str = "mean") -> float | np.ndarray:

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().numpy()
    if isinstance(y_lower, torch.Tensor):
        y_lower = y_lower.detach().numpy()
    if isinstance(y_upper, torch.Tensor):
        y_upper = y_upper.detach().numpy()
    if reduction == "mean":
        picp = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        return picp
    elif reduction == "all":
        picp = (y_true >= y_lower) & (y_true <= y_upper)
        return np.mean(picp, axis=(1, 2))


def eval_mpiw(y_lower: torch.Tensor | np.ndarray,
              y_upper: torch.Tensor | np.ndarray,
              reduction: str = "mean") -> float | np.ndarray:

    if isinstance(y_lower, torch.Tensor):
        y_lower = y_lower.detach().numpy()
    if isinstance(y_upper, torch.Tensor):
        y_upper = y_upper.detach().numpy()

    if reduction == "mean":
        mpiw = np.mean(y_upper - y_lower)
        return mpiw
    elif reduction == "all":
        mpiw = y_upper - y_lower
        return np.mean(mpiw, axis=(1, 2))
    else:
        raise ValueError("reduction must be mean or all")


### new functions for regression ECE and reliability diagrams
def get_prediction_interval_from_var(
    y_pred: torch.Tensor | np.ndarray,
    y_var: torch.Tensor | np.ndarray,
    alpha: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct Gaussian prediction intervals from mean and variance.

    Parameters
    ----------
    y_pred : torch.Tensor or np.ndarray
        Predicted mean.
    y_var : torch.Tensor or np.ndarray
        Predicted variance.
    alpha : float, optional
        Nominal coverage level, by default 0.95.

    Returns
    -------
    y_lower : np.ndarray
        Lower prediction bound.
    y_upper : np.ndarray
        Upper prediction bound.
    """

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_var, torch.Tensor):
        y_var = y_var.detach().cpu().numpy()

    y_std = np.sqrt(np.maximum(y_var, 1e-12))
    z = NormalDist().inv_cdf(0.5 + alpha / 2.0)

    y_lower = y_pred - z * y_std
    y_upper = y_pred + z * y_std

    return y_lower, y_upper


def eval_ece_regression(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    y_var: torch.Tensor | np.ndarray,
    alphas: list[float] | np.ndarray = None,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Evaluate regression ECE using interval coverage.

    This extends PICP to multiple confidence levels:
        ECE_reg = (1/K) * sum_k | C_hat(alpha_k) - alpha_k |

    Parameters
    ----------
    y_true : torch.Tensor or np.ndarray
        Ground-truth target.
    y_pred : torch.Tensor or np.ndarray
        Predicted mean.
    y_var : torch.Tensor or np.ndarray
        Predictive variance, can be total uncertainty or just epistemic uncertainty.
    alphas : list[float] or np.ndarray, optional
        Confidence levels. Default: np.arange(0.1, 1.0, 0.1)
    reduction : str, optional
        "mean" returns scalar ECE,
        "all" returns per-alpha absolute calibration gaps.

    Returns
    -------
    float or np.ndarray
        Regression ECE or per-alpha absolute gaps.
    """

    if alphas is None:
        alphas = np.arange(0.1, 1.0, 0.1)

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_var, torch.Tensor):
        y_var = y_var.detach().cpu().numpy()

    gaps = []

    for alpha in alphas:
        y_lower, y_upper = get_prediction_interval_from_var(y_pred, y_var, alpha)
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        gap = abs(coverage - alpha)
        gaps.append(gap)

    gaps = np.asarray(gaps)

    if reduction == "mean":
        return float(np.mean(gaps))
    elif reduction == "all":
        return gaps
    else:
        raise ValueError("reduction must be 'mean' or 'all'")


def get_reliability_curve_regression(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    y_var: torch.Tensor | np.ndarray,
    alphas: list[float] | np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return points for regression reliability diagram.

    Parameters
    ----------
    y_true : torch.Tensor or np.ndarray
        Ground-truth target.
    y_pred : torch.Tensor or np.ndarray
        Predicted mean.
    y_var : torch.Tensor or np.ndarray
        Predictive variance.
    alphas : list[float] or np.ndarray, optional
        Confidence levels. Default: np.arange(0.1, 1.0, 0.1)

    Returns
    -------
    nominal : np.ndarray
        Nominal confidence levels.
    empirical : np.ndarray
        Empirical coverages.
    """

    if alphas is None:
        alphas = np.arange(0.1, 1.0, 0.1)

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_var, torch.Tensor):
        y_var = y_var.detach().cpu().numpy()

    empirical = []

    for alpha in alphas:
        y_lower, y_upper = get_prediction_interval_from_var(y_pred, y_var, alpha)
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        empirical.append(coverage)

    return np.asarray(alphas), np.asarray(empirical)