# ------------------ Beginning of Reference Python Module ---------------------
""" Module for loading the plasticity discovery dataset

Classes
-------
PlasticityLaw: Load plasticity dataset, which contains training data and ground
truth.

"""
#
#                                                                       Modules
# =============================================================================
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================

# define the format of the visualization
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
formatter.set_useMathText(True)


class PlasticityLaw:
    """Class that loads the noisy plasticity law dataset,
    """

    def __init__(self,
                 dataset_path: str = None,
                 ground_truth: bool = True,
                 ground_truth_data_path: str = None) -> None:
        """ Load plasticity dataset for Bayesian recurrent neural network

        Parameters
        ----------
        dataset_path : str
            path of the pickle file of the training dataset
        ground_truth: bool
            load the ground truth data or not
        ground_truth_data_path: str
            path of the ground truth dataset


        """
        # get the path of the repository
        self.file_path = Path(__file__).parent.parent.as_posix()
        # filename
        dataset_file_name = self.file_path + "/problems/" + dataset_path
        # load the data
        self.dataset: pd.DataFrame = pd.read_pickle(dataset_file_name)
        # number of samples in dataset
        self.num_samples = len(self.dataset)

        if ground_truth:
            ground_truth_file_name = \
                self.file_path + "/problems/" + ground_truth_data_path
            self.ground_truth: pd.DataFrame = pd.read_pickle(
                ground_truth_file_name)

            # number of samples in test dataset (ground truth)
            self.num_ground_truth_samples = len(self.ground_truth)
        else:
            print("No ground truth data is provided.")

    def get_train_val_split(self,
                            num_train: int,
                            num_val: int,
                            seed: int = 1) -> None:
        """Get the processed data for training, validation, and testing

        Parameters
        ----------
        num_train : int
            Number of training data
        num_val : int
            Number of validation data
        seed: int
            seed of selecting the training paths
        """
        # Convert to torch tensors
        self.X, self.Y = self.convert_data_to_torch()
        self.scale_dataset()

        total_samples = len(self.dataset)
        requested_total = num_train + num_val

        if requested_total > total_samples:
            raise ValueError(
                "Requested split exceeds the total number of samples.")

        # Shuffle the indices with certain seed
        indices = torch.randperm(
            total_samples, generator=torch.Generator().manual_seed(seed))

        # Compute split boundaries
        train_end = num_train
        val_end = train_end + num_val

        # Assign data splits using non-overlapping indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]

        self.strain_train = self.strain_normalized[train_indices]
        self.strain_validate = self.strain_normalized[val_indices]

        self.stress_train = self.stress_normalized[train_indices]
        self.stress_validate = self.stress_normalized[val_indices]


    def convert_data_to_torch(self) -> Tuple[Tensor, Tensor]:
        """convert the data to torch format

        Returns
        -------
        Tuple[Tensor, Tensor]
            strains and stresses in torch format
        """
        # empty lists for data
        X, Y = [], []
        # gen number of samples
        num_samples = len(self.dataset)
        # get the data
        for ii in range(num_samples):
            X.append(
                (torch.FloatTensor(self.dataset['strain'].iloc[ii]).flatten(
                    start_dim=1)[:, [0, 1, 3]]).unsqueeze(0))
            Y.append(
                (torch.FloatTensor(self.dataset['stress'].iloc[ii]).flatten(
                    start_dim=1)[:, [0, 1, 3]]).unsqueeze(0))
        # concatenate all date together
        X, Y = torch.cat(X, dim=0), torch.cat(Y, dim=0)

        return X, Y

    def scale_dataset(self) -> None:
        """scale the dataset
        """
        self.strain_normalized, self.strain_mean, self.strain_std = \
            self._normalize_data(
                data=self.X)
        self.stress_normalized, self.stress_mean, self.stress_std = \
            self._normalize_data(
                data=self.Y)

    def get_ground_truth(self) -> None:
        """get the processed data for testing
        """
        if not hasattr(self, 'ground_truth'):
            print("No ground truth data is provided.")
            return
        else:
            # first convert to torch
            x_test = []
            y_test_mean = []
            y_test_std = []
            for ii in range(self.num_ground_truth_samples):
                x_test.append((
                    torch.FloatTensor(
                        self.ground_truth['strain_mean'].iloc[ii]).flatten(
                        start_dim=1)[:, [0, 1, 3]]
                ).unsqueeze(0))
                y_test_mean.append((
                    torch.FloatTensor(
                        self.ground_truth['stress_mean'].iloc[ii]).flatten(
                        start_dim=1)[:, [0, 1, 3]]
                ).unsqueeze(0))
                y_test_std.append((
                    torch.FloatTensor(
                        self.ground_truth['stress_std'].iloc[ii]).flatten(
                        start_dim=1)[:, [0, 1, 3]]
                ).unsqueeze(0))

            # original scale
            self.strain_ground_truth = torch.cat(x_test, dim=0)
            self.stress_ground_truth_mean = torch.cat(y_test_mean, dim=0)
            self.stress_ground_truth_std = torch.cat(y_test_std, dim=0)

            # scale the data
            self.strain_ground_truth_normalized = (
                self.strain_ground_truth - self.strain_mean) / self.strain_std
            self.stress_ground_truth_mean_normalized = (
                self.stress_ground_truth_mean - self.stress_mean) / self.stress_std
            self.stress_ground_truth_std_normalized = (
                self.stress_ground_truth_std) / self.stress_std
            

    def plot_training_data(self,
                           index: int,
                           save_figure: bool = False) -> None:
        """plot the training data

        index: int
            index of the path
        save_figure: bool
            save the figure to disk or not
        """

        # get the data
        fig, ax = plt.subplots(2, 3, figsize=(12, 5))
        pparam = dict(ylabel=r"$E_{11}$")
        ax[0, 0].plot(self.dataset["strain"][index][:, 0, 0],
                      color="#0077BB",
                      linewidth=2)
        ax[0, 0].set(**pparam)
        # set the limits of the y axis
        pparam = dict(ylabel=r"$E_{12}$")
        ax[0, 1].plot(self.dataset["strain"][index][:, 0, 1],
                      color="#0077BB",
                      linewidth=2)
        ax[0, 1].set(**pparam)
        pparam = dict(ylabel=r"$E_{22}$")
        ax[0, 2].plot(self.dataset["strain"][index][:, 1, 1],
                      color="#0077BB",
                      linewidth=2)
        ax[0, 2].set(**pparam)
        pparam = dict(xlabel="Time step", ylabel=r"$\sigma_{11}$ (MPa)")
        ax[1, 0].plot(self.dataset["stress"][index][:, 0, 0],
                      color="#0077BB",
                      linewidth=2)

        ax[1, 0].set(**pparam)
        pparam = dict(xlabel="Time step", ylabel=r"$\sigma_{12}$ (MPa)")
        ax[1, 1].plot(self.dataset["stress"][index][:, 0, 1],
                      color="#0077BB",
                      linewidth=2)

        ax[1, 1].set(**pparam)
        pparam = dict(xlabel="Time step", ylabel=r"$\sigma_{22}$ (MPa)")
        ax[1, 2].plot(self.dataset["stress"][index][:, 1, 1],
                      color="#0077BB",
                      linewidth=2)
        ax[1, 2].set(**pparam)
        # set the fontsize of the axes
        for i in range(2):
            for j in range(3):
                ax[i, j].tick_params(axis='both', which='major', labelsize=12)
                ax[i, j].tick_params(axis='both', which='minor', labelsize=12)
        # set the linewidth of the axes
        for i in range(2):
            for j in range(3):
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax[i, j].spines[axis].set_linewidth(1.5)
        # set the fontsize of the labels
        for i in range(2):
            for j in range(3):
                ax[i, j].set_xlabel(ax[i, j].get_xlabel(), fontsize=14)
                ax[i, j].set_ylabel(ax[i, j].get_ylabel(), fontsize=14)
                ax[i, j].yaxis.set_major_formatter(
                    formatter)
        # adjust the space between the subplots
        plt.subplots_adjust(wspace=0.32, hspace=0.25)
        # save the figure
        if save_figure:
            plt.savefig(f"train_data_{index}.png",
                        dpi=300, bbox_inches="tight")
            plt.savefig(f"train_data_{index}.svg",
                        dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def plot_test_data(self,
                       index: int,
                       save_figure: bool = False) -> None:
        """plot the data

        Parameters
        ----------
        index : int
            index of the data
        save_figure : bool, optional
            save the figure, by default False

        """

        # get the data
        rve_data_strain_mean = self.ground_truth["strain_mean"][index]

        rve_data_stress_mean = self.ground_truth["stress_mean"][index]
        rve_data_stress_std = self.ground_truth["stress_std"][index]

        # length of the data
        length_of_strain = rve_data_strain_mean.shape[0]

        fig, ax = plt.subplots(2, 3, figsize=(12, 5))
        # set the title of the whole figure
        pparam = dict(ylabel=r"$E_{11}$")
        ax[0, 0].plot(rve_data_strain_mean[:, 0, 0],
                      color="#0077BB",
                      linewidth=2,
                      label="rve")
        ax[0, 0].set(**pparam)
        # set the limits of the y axis
        pparam = dict(ylabel=r"$E_{12}$")
        ax[0, 1].plot(rve_data_strain_mean[:, 0, 1],
                      color="#0077BB",
                      linewidth=2,
                      label="rve")

        ax[0, 1].set(**pparam)
        pparam = dict(ylabel=r"$E_{22}$")
        ax[0, 2].plot(rve_data_strain_mean[:, 1, 1],
                      color="#0077BB",
                      linewidth=2,
                      label="rve")

        ax[0, 2].set(**pparam)
        pparam = dict(xlabel="Time step", ylabel=r"$\sigma_{11}$ (MPa)")

        ax[1, 0].plot(rve_data_stress_mean[:, 0, 0],
                      color="#0077BB",
                      linewidth=2,
                      label="rve")
        ax[1, 0].fill_between(np.arange(length_of_strain),
                              rve_data_stress_mean[:, 0, 0] -
                              1.96 * rve_data_stress_std[:, 0, 0],
                              rve_data_stress_mean[:, 0, 0] +
                              1.96*rve_data_stress_std[:, 0, 0],
                              color="#0077BB",
                              edgecolor="none",
                              alpha=0.5)

        ax[1, 0].set(**pparam)

        pparam = dict(xlabel="Time step", ylabel=r"$\sigma_{12}$ (MPa)")

        ax[1, 1].plot(rve_data_stress_mean[:, 0, 1],
                      color="#0077BB",
                      linewidth=2,
                      label="rve")
        ax[1, 1].fill_between(np.arange(length_of_strain),
                              rve_data_stress_mean[:, 0, 1] -
                              1.96*rve_data_stress_std[:, 0, 1],
                              rve_data_stress_mean[:, 0, 1] +
                              1.96*rve_data_stress_std[:, 0, 1],
                              color="#0077BB",
                              edgecolor="none",
                              alpha=0.5)
        ax[1, 1].set(**pparam)

        pparam = dict(xlabel="Time step", ylabel=r"$\sigma_{22}$ (MPa)")

        ax[1, 2].plot(rve_data_stress_mean[:, 1, 1],
                      color="#0077BB",
                      linewidth=2,
                      label="RVE")
        ax[1, 2].fill_between(np.arange(length_of_strain),
                              rve_data_stress_mean[:, 1, 1] -
                              1.96*rve_data_stress_std[:, 1, 1],
                              rve_data_stress_mean[:, 1, 1] +
                              1.96 * rve_data_stress_std[:, 1, 1],
                              color="#0077BB",
                              edgecolor="none",
                              alpha=0.5)

        ax[1, 2].set(**pparam)
        ax[1, 2].legend([
            "Ground Truth Mean",
            "Ground Truth 95% CI"
        ], loc="best", fontsize=8, edgecolor="none", frameon=False)

        for i in range(2):
            for j in range(3):
                ax[i, j].tick_params(axis='both', which='major', labelsize=12)
                ax[i, j].tick_params(axis='both', which='minor', labelsize=12)
        # set the linewidth of the axes
        for i in range(2):
            for j in range(3):
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax[i, j].spines[axis].set_linewidth(1.5)
        for i in range(2):
            for j in range(3):
                ax[i, j].set_xlabel(ax[i, j].get_xlabel(), fontsize=14)
                ax[i, j].set_ylabel(ax[i, j].get_ylabel(), fontsize=14)
                ax[i, j].yaxis.set_major_formatter(
                    formatter)
        # adjust the space between the subplots
        plt.subplots_adjust(wspace=0.32, hspace=0.25)
        # save the figure
        if save_figure:
            plt.savefig(f"test_data_{index}.svg", dpi=300, bbox_inches="tight")
        else:
            plt.show()

    @staticmethod
    def _normalize_data(data) -> Tuple[Tensor, Tensor, Tensor]:
        """normalize the dataset

        Parameters
        ----------
        data : _type_
            _description_

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            _description_
        """
        dim = (0, 1)
        data_mean = data.mean(dim=dim, keepdim=True)
        data_std = data.std(dim=dim, unbiased=False, keepdim=True)

        data_normalized = (data - data_mean) / data_std

        return data_normalized, data_mean, data_std
