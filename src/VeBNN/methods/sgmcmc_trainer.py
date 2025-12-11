# ------------------ Beginning of Reference Python Module ---------------------
""" This module contains the class BayesSampler, which is a approximate
Bayesian inference class for sampling the posterior of the Bayesian neural
network.

Class:
    SFTrainer: Abstract class for training single-fidelity deep neural networks
    and single-fidelity Bayesian neural networks.

"""
#
#                                                                       Modules
# =============================================================================
import copy
from typing import Tuple
import os
import shutil
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, random_split, DataLoader
from typing import Dict

# local imports
from ..samplers import pSGLD, SGHMC, SGLD
from .loss import GammaNLLLoss, NLLLoss
from ..networks import MeanNet, GammaVarNet


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================



class SGMCMCTrainer:
    """A wrapper class for the VeBNN training and evaluating its own
    performance.
    """

    def __init__(
        self,
        mean_net: MeanNet,
        var_net: GammaVarNet,
        device: torch.device = torch.device("cpu"),
        job_id: int = 1,
    ) -> None:
        """initialize the StandardRNN class, with RNN architecture, device and
        seed.

        Parameters
        ----------
        mean_net : MeanNet
            the mean network architecture used for VeBNN
        var_net : GammaVarNet
            the variance network architecture used for VeBNN
        device : torch.device, optional
            device with cpu or gpu, by default torch.device("cpu")
        """

        # set device
        self.device = device
        # Model architecture of mean and variance networks
        self.un_trained_mean_net = mean_net.to(self.device)
        self.un_trained_var_net = var_net.to(self.device)
        # get the job id
        self.job_id = job_id


    def cooperative_train(self,
                          x_train: torch.Tensor,
                          y_train: torch.Tensor,
                          iteration: int,
                          init_config: Dict = {"loss_name": "MSE",
                                               "optimizer_name": "Adam",
                                               "lr": 1e-3,
                                               "weight_decay": 1e-6,
                                               "num_epochs": 1000,
                                               "batch_size": 32,
                                               "verbose": True,
                                               "print_iter": 100, },
                          var_config: Dict = {"optimizer_name": "Adam",
                                              "lr": 1e-3,
                                              "num_epochs": 1000,
                                              "batch_size": 32,
                                              "verbose": True,
                                              "print_iter": 100,
                                              "early_stopping": True,
                                              "early_stopping_iter": 100,
                                              "early_stopping_tol": 1e-4, },
                          sampler_config: Dict = {"sampler": "pSGLD",
                                                  "lr": 1e-3,
                                                  "gamma": 0.9999,
                                                  "num_epochs": 1000,
                                                  "mix_epochs": 100,
                                                  "burn_in_epochs": 100,
                                                  "batch_size": 32,
                                                  "verbose": True,
                                                  "print_iter": 100, },
                          delete_model_raw_data: bool = True,
                          ) -> None:
        """Train the model using cooperative training."""
        # convert the data to the device
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        # Initialize the model
        self.warm_up_mean_net, _, _ = self._initialization(
                x=x_train,
                y=y_train,
                init_config=init_config,
            )
    
        # begin the second and third training
        mar_likelihoods = []
        # adding an array for recording the results
        for ii in range(iteration):
            print("=========================================================")
            print(
                f"Step 2: Train for the variance network, iteration {ii+1}")
            self.configure_var_optimizer(
                var_net=None,
                optimizer_name=var_config["optimizer_name"],
                lr=var_config["lr"],)
            self.var_train(
                x_train=x_train,
                y_train=y_train,
                num_epochs=var_config["num_epochs"],
                batch_size=int(np.min(
                    [var_config["batch_size"], x_train.shape[0]])),
                early_stopping=var_config["early_stopping"],
                early_stopping_iter=var_config["early_stopping_iter"],
                early_stopping_tol=var_config["early_stopping_tol"],
                verbose=var_config["verbose"],
                print_iter=var_config["print_iter"],
                iteration=ii)
            # get the aleatoric uncertainty for the training dataset
            aleatoric_var = self.aleatoric_variance_predict(x=x_train)
            print("Finished training the variance network")
            print("===========================================")
            print("Step 3: Train for the mean network with SGMCMC")
            # update the mean with MCMC sampler
            self.configure_bayes_sampler(
                    mean_net=self.warm_up_mean_net,
                    sampler=sampler_config["sampler"],
                    lr=sampler_config["lr"],)
            # begin to sample the posterior
            self.sample_posterior(
                x=x_train,
                y=y_train,
                var_best=aleatoric_var,
                num_epochs=sampler_config["num_epochs"],
                burn_in_epochs=sampler_config["burn_in_epochs"],
                mix_epochs=sampler_config["mix_epochs"],
                batch_size=int(np.min(
                    [sampler_config["batch_size"], x_train.shape[0]])),
                verbose=sampler_config["verbose"],
                print_iter=sampler_config["print_iter"])

            # get the ppd
            _, _ = self.bayes_predict(x_train, save_ppd=True)
            # get the log marginal likelihood
            lmglk = self.log_marginal_likelihood(
                y_train,
                var_best=None,
                refinement="mean")
            mar_likelihoods.append(lmglk)

            print("Finished training the Bayesian mean network")
            print("============================================")

            if os.path.exists(f"model_data_{self.job_id}") is False:
                os.makedirs(f"model_data_{self.job_id}")
                print("Create model data folder to save the temporary models")
            # save the model
            torch.save(self.mean_nets,
                       f"model_data_{self.job_id}/mean_net_iter_{ii}.pth")
            torch.save(self.best_var_net,
                       f"model_data_{self.job_id}/var_net_iter_{ii}.pth")
        if iteration == 1:
            self.best_mgk_idx = 0
        else:
            self.best_mgk_idx = np.argmax(mar_likelihoods[1:]) + 1
            self.best_log_mgk = mar_likelihoods[self.best_mgk_idx]
        print("=========================================================")
        print("Finished training the model")

        # get the best model
        self.mean_nets = torch.load(
            f"model_data_{self.job_id}/mean_net_iter_{self.best_mgk_idx}.pth")
        self.best_var_net = torch.load(
            f"model_data_{self.job_id}/var_net_iter_{self.best_mgk_idx}.pth",
            weights_only=False)
        if delete_model_raw_data:
            if os.path.exists(f"model_data_{self.job_id}"):
                shutil.rmtree(f"model_data_{self.job_id}")
            print("Delete the model data folder to free space")
        self.best_log_mgk = mar_likelihoods[self.best_mgk_idx]

    def _initialization(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        init_config: Dict = {
                            "loss_name": "MSE",
                            "optimizer_name": "Adam",
                            "lr": 1e-3,
                            "weight_decay": 1e-6,
                            "num_epochs": 1000,
                            "batch_size": 32,
                            "verbose": True,
                            "print_iter": 100,
                            "split_ratio": 0.8,
                        },
                        ):
        """
        Initialize the mean network using MAP training.
        """
        self.warm_up_train_loss = []
        self.warm_up_val_loss = []
        # prepare for the data loader
        dataset = TensorDataset(x, y)
        n_total = len(dataset)

        # create train and validation split for the warm-up training
        split_ratio = init_config["split_ratio"]
        train_size = int(split_ratio * n_total)
        val_size = n_total - train_size
        # random split for getting train and validation datasets
        train_dataset, val_dataset = random_split(dataset,
                                                  [train_size, val_size])

        # get the batch size
        batch_size = int(
            np.min([init_config["batch_size"], train_size])
        )
        # define data loaders
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        # define the validation data loader
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False)


        # === copy from the untrained network ===
        map_net = copy.deepcopy(self.un_trained_mean_net)

        # === define optimizer ===
        if init_config["optimizer_name"] == "Adam":
            optimizer = torch.optim.Adam(
                map_net.parameters(),
                lr=init_config["lr"],
                weight_decay=init_config["weight_decay"]
            )
        elif init_config["optimizer_name"] == "SGD":
            optimizer = torch.optim.SGD(
                map_net.parameters(),
                lr=init_config["lr"],
                weight_decay=init_config["weight_decay"]
            )
        else:
            msg = f"Undefined optimizer: {init_config['optimizer_name']}"
            raise ValueError(msg)
        # === define data criterion ===
        loss_name = init_config["loss_name"].upper()
        if loss_name == "MSE":
            data_criterion = torch.nn.MSELoss(reduction="mean")
        else:
            msg = f"Unsupported loss: {loss_name}"
            raise ValueError(msg)

        # === training loop ===
        best_state_dict = copy.deepcopy(map_net.state_dict())
        best_epoch = 0
        best_val_loss = float("inf")

        for epoch in range(init_config["num_epochs"]):
            map_net.train()
            train_loss_epoch = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                y_pred = map_net(xb)
                loss = data_criterion(y_pred, yb)
                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.detach().item() * xb.size(0)  

            train_loss_epoch /= train_size
            self.warm_up_train_loss.append(train_loss_epoch)
            # === validation loss ===
            if val_loader is not None:
                map_net.eval()
                val_loss_epoch = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        y_pred = map_net(xb)
                        val_loss_epoch += data_criterion(y_pred, yb).item() * xb.size(0)
                val_loss_epoch /= val_size
            else:
                val_loss_epoch = train_loss_epoch 
            self.warm_up_val_loss.append(val_loss_epoch)
            if init_config["verbose"] and (
                (epoch + 1) % init_config["print_iter"] == 0 or epoch == 0
            ):
                print(
                    f"Epoch {epoch+1}/{init_config['num_epochs']}, "
                    f"Train loss: {train_loss_epoch:.3e}, "
                    f"Val loss: {val_loss_epoch:.3e}"
                )

            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                best_epoch = epoch
                best_state_dict = copy.deepcopy(map_net.state_dict())

        # load the best model
        map_net.load_state_dict(best_state_dict)      

        return map_net, best_epoch, best_val_loss

    def configure_var_optimizer(
        self,
        var_net: GammaVarNet = None,
        optimizer_name: str = "Adam",
        lr: float = 1e-3,
    ) -> None:
        """define optimizer of the network

        Parameters
        ----------
        optimizer_name : str, optional
            name of the optimizer, by default "Adam"
        lr : float, optional
            learning rate, by default 1e-3

        Raises
        ------
        ValueError
            Undefined optimizer
        """
        # take a copy from the untrained network
        if var_net is not None:
            self.var_net = copy.deepcopy(var_net)
        else:
            self.var_net = copy.deepcopy(self.un_trained_var_net)

        # define optimizer
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(
                self.var_net.parameters(),
                lr=lr,
            )
        elif optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(
                self.var_net.parameters(),
                lr=lr,
            )
        else:
            raise ValueError("Undefined optimizer")

    def configure_bayes_sampler(
        self,
        mean_net: MeanNet = None,
        sampler: str = "pSGLD",
        lr: float = 1e-3,
        gamma: float = 0.9999,
    ) -> None:
        """define optimizer and learning rate scheduler

        Parameters
        ----------
        lr : float
            learning rate, by default 1e-3
        gamma : float, optional
            learning rate decay, by default 0.9999
        """
        if mean_net is not None:
            self.mean_net = copy.deepcopy(mean_net)
        else:
            self.mean_net = copy.deepcopy(self.un_trained_mean_net)
        # define optimizer
        if sampler == "pSGLD":
            self.sampler = pSGLD(
                self.mean_net.parameters(),
                lr=lr
            )
        elif sampler == "SGHMC":
            self.sampler = SGHMC(
                self.mean_net.parameters(),
                lr=lr
            )
        elif sampler == "SGLD":
            self.sampler = SGLD(
                self.mean_net.parameters(),
                lr=lr
            )
        else:
            raise ValueError("Undefined inference method")

        # define learning rate scheduler
        self.scheduler = ExponentialLR(self.sampler, gamma=gamma)

    def sample_posterior(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        var_best: torch.Tensor,
        num_epochs: int,
        mix_epochs: int,
        burn_in_epochs: int,
        batch_size: int = None,
        verbose: bool = True,
        print_iter: int = 10,
    ) -> None:

        if batch_size is None: batch_size = x.size(0)

        dataset = TensorDataset(x, y, var_best)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True)
        # initialize the storage for posterior samples
        self.mean_nets = [] 
        self.log_likelihood = []
        self.log_prior = []
        # set the mean network to training mode
        self.mean_net.train()
        for epoch in range(num_epochs):
            nll_loss_collection = 0.0
            neg_log_prior_collection = 0.0
            for X_batch, y_batch, var_batch in dataloader:

                X_batch, y_batch, var_batch = (
                    X_batch.to(self.device),
                    y_batch.to(self.device),
                    var_batch.to(self.device),
                )
                self.sampler.zero_grad()
                pred = self.mean_net(X_batch)
                nll_loss = NLLLoss()(pred=pred, 
                                     pred_var=var_batch,
                                     real=y_batch,
                                      num_scale=len(dataloader))
                prior_loss = self.mean_net.neg_log_prior()
                loss = nll_loss + prior_loss
                loss.backward()
                self.sampler.step()
                nll_loss_collection += nll_loss.detach()
                neg_log_prior_collection += prior_loss.detach()

            self.scheduler.step()

            if verbose and (epoch + 1) % print_iter == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"NLL: {nll_loss_collection.item():.3e}, "
                    f"Neg log prior: {neg_log_prior_collection.item():.3e}"
                )

            if epoch >= burn_in_epochs and (epoch % mix_epochs == 0):
                self.mean_nets.append(
                    copy.deepcopy(self.mean_net.state_dict()))
                self.log_likelihood.append(-nll_loss_collection)
                self.log_prior.append(-neg_log_prior_collection)

    def var_train(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        num_epochs: int,
        batch_size: int,
        verbose: bool = False,
        print_iter: int = 100,
        early_stopping: bool = False,
        early_stopping_iter: float = 100,
        early_stopping_tol: float = 1e-4,
        iteration: int = 0,
    ):
        """Train the variance network."""
        min_loss = float("inf")
        # check if we have self.nets or not
        if iteration == 0:
            train_mean = self._warm_up_predict(x_train)
        else:
            # get the ppd responses
            train_mean, _ = self.bayes_predict(x_train, save_ppd=True)
        # get the residuals for the training dataset
        residuals = (y_train - train_mean)**2

        # split the data into batches
        dataset = TensorDataset(x_train, residuals, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.train_loss_collection = torch.zeros(num_epochs)
        self.nlog_mglks = torch.zeros(num_epochs)
        # count the number of epochs with no improvement
        if early_stopping: no_improvement = 0

        # begin the training process
        for epoch in range(num_epochs):
            # set the network to training mode
            self.var_net.train()
            log_mglks_batch = 0.0
            num_sample_count = 0
            for i, (x_batch, residuals_batch,y_batch ) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                residuals_batch = residuals_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                alpha, beta = self.var_net.forward(x_batch)
                # get the mean and variance of the prediction and add penalty
                nll_loss = GammaNLLLoss(reduction="sum")(
                    residuals=residuals_batch,
                    alpha=alpha,
                    beta=beta,
                    num_scale=len(dataloader))
                # get the prior loss
                prior_loss = self.var_net.neg_log_prior()
                loss = nll_loss + prior_loss
                # update used samples
                num_batch_sample = x_batch.shape[0]
                if iteration > 0:
                    # update the log_margin_likelihood for this batch
                    log_marginal_likelihood = self.log_marginal_likelihood(
                        y=y_batch,
                        ppd_responses=self.responses[:, num_sample_count:(
                            num_sample_count+num_batch_sample), ...],
                        refinement="var",
                        var_best=alpha/beta)

                    # sum over the mini-batch
                    log_mglks_batch += log_marginal_likelihood*num_batch_sample
                    num_sample_count += num_batch_sample

                loss.backward()
                self.optimizer.step()

            self.train_loss_collection[epoch] = loss.item()
            # print to screen
            if verbose and epoch % print_iter == 0:
                print(
                    f"Epoch/Total: {epoch}/{num_epochs}, "
                    f"Gamma NLL: {loss.item():.3e}, "
                    f"neg log prior: {prior_loss.item():.3e}, "
                    f"log marginal likelihood: {log_mglks_batch/x_train.shape[0]:.3e}"
                )
            #  check early stopping
            if iteration > 0:
                self.nlog_mglks[epoch] = -log_mglks_batch/x_train.shape[0]
                if early_stopping:
                    if epoch > 0:
                        curr = float(self.nlog_mglks[epoch])
                        rel_improvement = abs(curr - min_loss) / abs(min_loss)
                    else:
                        curr = float(self.nlog_mglks[epoch])
                        rel_improvement = 1.0
                    if rel_improvement > early_stopping_tol and curr < min_loss:
                        no_improvement = 0
                        min_loss = curr
                        self.best_var_epoch = epoch
                        self.best_var_net = copy.deepcopy(self.var_net)
                    else:
                        no_improvement += 1
                        if no_improvement >= early_stopping_iter:
                            break
                else:
                    # update the best model no early stopping
                    self.best_var_epoch = epoch
                    self.best_var_net = copy.deepcopy(self.var_net)
            else:
                # update the best model no early stopping
                self.best_var_epoch = epoch
                self.best_var_net = copy.deepcopy(self.var_net)

        if hasattr(self, "responses"):
            del self.responses
            print(
                "Training complete. "
                "Deleting temporary PPD responses to free memory"
            )

        return self.best_var_net, self.best_var_epoch

    def log_marginal_likelihood(self,
                                y: torch.Tensor,
                                var_best: torch.Tensor = None,
                                ppd_responses: torch.Tensor = None,
                                refinement: str = "mean",
                                ) -> float:
        """
        Evaluation of log marginal likelihood

        Parameters
        ----------
        y : torch.Tensor
            Target values (ground truth).
        var_best : torch.Tensor
            Precomputed aleatoric variance from the best model.
        refinement : str, optional
            Refinement method for log marginal likelihood, by default "mean".

        Returns
        -------
        float
            Log marginal likelihood.
        """
        if refinement == "var":
            # make sure the ppd responses are available
            if ppd_responses is None:
                raise ValueError("PPD responses are not available")
            
            # Compute negative log-likelihood (NLL) for all samples in parallel
            # Shape: (num_samples, batch_size, output_dim)
            residuals = (ppd_responses - y.unsqueeze(0))
            nlls = 0.5 * torch.sum(residuals**2 / var_best.unsqueeze(0) +
                                   torch.log(var_best.unsqueeze(0)), dim=-1)

            # get the kl divergence
            # kl_values = torch.stack(self.kl_values)
            log_likelihoods = - torch.sum(nlls, dim=-1)  # + kl_values
            max_log = torch.max(log_likelihoods)
            log_marginal_likelihood = max_log + \
                torch.log(torch.mean(torch.exp(log_likelihoods - max_log)))

        elif refinement == "mean":
            # Combine log-likelihoods and log-priors (KL values)
            log_posterior_values = torch.stack(
                self.log_likelihood) + torch.stack(self.log_prior)
            # Use log-sum-exp trick for numerical stability
            max_log = torch.max(log_posterior_values)
            log_marginal_likelihood = max_log + torch.log(
                torch.mean(torch.exp(log_posterior_values - max_log))
            )

        return log_marginal_likelihood.item()

    def aleatoric_variance_predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Set the model to evaluation mode
        self.best_var_net.eval()
        with torch.no_grad():
            # Forward pass through the
            # variance network to get the predicted variance
            alpha, beta = self.best_var_net(x.to(self.device))
            # Compute the predicted variance
            var_pred = alpha / beta

        return var_pred.detach()

    def bayes_predict(
        self,
        x: torch.Tensor,
        save_ppd: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the mean and variance of the output at the scaled data.

        Parameters
        ----------
        x : torch.Tensor
            Test data points.
        save_ppd : bool
            Whether to save posterior predictive distributions.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Predicted mean and variance at the scaled space.
        """
        responses = []

        for state_dict in self.mean_nets:
            # Create a fresh model instance
            temp_model = copy.deepcopy(self.mean_net)
            temp_model.load_state_dict(state_dict)
            temp_model.to(self.device)
            temp_model.eval()
            with torch.no_grad():
                y_pred = temp_model.forward( x.to(self.device))
                responses.append(y_pred)

        # Stack the predictions and calculate the mean and variance
        # Shape: (num_samples, batch_size, output_dim)
        responses = torch.stack(responses)
        y_pred_mean = torch.mean(responses, dim=0)
        y_pred_var = torch.var(responses, dim=0)
        if save_ppd:
            self.responses = responses
        return y_pred_mean.detach(), y_pred_var.detach()

    def _warm_up_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using the warm-up mean network.

        Parameters
        ----------
        x : torch.Tensor
            Test data points.
        Returns
        -------
        Tensor
            Predicted mean at the scaled space.
        """
        if not hasattr(self, "warm_up_mean_net"):
            raise ValueError(
                "Warm-up mean network not found. "
                "Please run initialization first."
            )
        self.warm_up_mean_net.eval()
        with torch.no_grad():
            y_pred = self.warm_up_mean_net(x.to(self.device))
        return y_pred.detach()



# ------------------ End of Reference Python Module ------------------------- #
