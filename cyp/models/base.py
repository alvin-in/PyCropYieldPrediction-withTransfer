import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
from tqdm import tqdm
from datetime import datetime
from .gp_rbf import GaussianProcess
from .loss import l1_l2_loss
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import optuna
import wandb


class ModelBase:
    """
    Base class for all models
    """

    total_epoch = 0

    def __init__(
            self,
            model,
            model_weight,
            model_bias,
            model_type,
            savedir,
            use_gp=True,
            sigma=1,
            r_loc=0.5,
            r_year=1.5,
            sigma_e=0.32,
            sigma_b=0.01,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):

        self.savedir = savedir / model_type
        self.savedir.mkdir(parents=True, exist_ok=True)

        print(f"Using {device.type}")
        if device.type != "cpu":
            model = model.cuda()
        self.model = model
        self.model_type = model_type
        self.model_weight = model_weight
        self.model_bias = model_bias

        self.device = device

        # for reproducability
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.gp = None
        if use_gp:
            self.gp = GaussianProcess(sigma, r_loc, r_year, sigma_e, sigma_b)

    def run(
            self,
            path_to_histogram=Path("data/arg_im_out/histogram_all_full.npz"),
            times="all",
            pred_years=None,
            num_runs=2,
            train_steps=25000,
            batch_size=32,
            starter_learning_rate=1e-3,
            weight_decay=0,
            l1_weight=0,
            patience=10,
            ret=False,
            multi_hyp=None,
            kfold=False
    ):
        """
        Train the models. Note that multiple models are trained: as per the paper, a model
        is trained for each year, with all preceding years used as training values. In addition,
        for each year, 2 models are trained to account for random initialization.
        Parameters
        ----------
        path_to_histogram: pathlib Path, default=Path('data/img_output/histogram_all_full.npz')
            The location of the training data
        times: {'all', 'realtime'}
            Which time indices to train the model on. If 'all', a full run (32 timesteps) is used.
            If 'realtime', range(10, 31, 4) is used.
        pred_years: int, list or None, default=None
            Which years to build models for. If None, the default values from the paper (range(2009, 2016))
            are used.
        num_runs: int, default=2
            The number of runs to do per year. Default taken from the paper
        train_steps: int, default=25000
            The number of steps for which to train the model. Default taken from the paper.
        batch_size: int, default=32
            Batch size when training. Default taken from the paper
        starter_learning_rate: float, default=1e-3
            Starter learning rate. Note that the learning rate is divided by 10 after 2000 and 4000 training
            steps. Default taken from the paper
        weight_decay: float, default=1
            Weight decay (L2 regularization) on the model weights
        l1_weight: float, default=0
            In addition to MSE, L1 loss is also used (sometimes). This is the weight to assign to this L1 loss.
        patience: int or None, default=10
            The number of epochs to wait without improvement in the validation loss before terminating training.
            Note that the original repository doesn't use early stopping.
        ret: boolean, default=False,
            If True, method returns error. Use for Optuna optimization.
        multi_hyp path, default=None,
            If not none: Path to csv with hyperparameter for each year.
        """
        wandb.config = {"learning_rate": starter_learning_rate, "train_steps": train_steps, "batch_size": batch_size}

        with np.load(path_to_histogram) as hist:
            images = hist["output_image"]
            locations = hist["output_locations"]
            yields = hist["output_yield"]
            years = hist["output_year"]
            indices = hist["output_index"]

        # to collect results
        years_list, run_numbers, rmse_list, me_list, times_list, r_sq_list = [], [], [], [], [], []
        if self.gp is not None:
            rmse_gp_list, me_gp_list, r_sq_gp_list = [], [], []

        if pred_years is None:
            pred_years = range(2010, 2020)  # standard 2009 bis 2016
        elif type(pred_years) is int:
            pred_years = [pred_years]

        if times == "all":
            times = [32]
        else:
            times = range(10, 31, 4)

        dropout = None

        wandb.define_metric("years")

        if multi_hyp is not None:
            print("multi_hyp training overwrittes given params!")

        for pred_year in pred_years:
            if multi_hyp is not None:
                df = pd.read_csv(multi_hyp)
                # learn_rate_decay = [df[df['year'] == pred_year].learn_rate_decay1,
                #                     df[df['year'] == pred_year].learn_rate_decay2]
                # num_runs = df[df['year'] == pred_year].num_runs
                train_steps = df[df['year'] == pred_year]['train_steps'].values[0]
                starter_learning_rate = df[df['year'] == pred_year]['learning_rate'].values[0]
                weight_decay = df[df['year'] == pred_year]['weighty_decay'].values[0]
                # l1_weight = df[df['year'] == pred_year].l1_weight
                # patience = df[df['year']==pred_year].patience
                # freeze = df[df['year']==pred_year].freeze
                # sp_weight = df[df['year'] == pred_year].sp_weight
                # l2_sp_beta = df[df['year'] == pred_year].l2_sp_beta
                # bss = df[df['year'] == pred_year].bss
                # bss_k = df[df['year'] == pred_year].bss_k
                # patience = df[df['year'] == pred_year]['patience'].values[0]
                dropout = df[df['year'] == pred_year]['dropout'].values[0]

            for run_number in range(1, num_runs + 1):
                for time in times:
                    print(
                        f"Training to predict on {pred_year}, Run number {run_number}"
                    )

                    results = self._run_1_year(
                        images,
                        yields,
                        years,
                        locations,
                        indices,
                        pred_year,
                        time,
                        run_number,
                        train_steps,
                        batch_size,
                        starter_learning_rate,
                        weight_decay,
                        l1_weight,
                        patience,
                        ret,
                        kfold,
                        dropout
                    )

                    years_list.append(pred_year)
                    run_numbers.append(run_number)
                    times_list.append(time)

                    if self.gp is not None:
                        rmse, me, rmse_gp, me_gp, r_sq, r_sq_gp = results
                        rmse_gp_list.append(rmse_gp)
                        me_gp_list.append(me_gp)
                        r_sq_gp_list.append(r_sq_gp)
                        # wandb.log({"rmse_gp": rmse_gp, "me_gp": me_gp}, step=pred_year)
                    else:
                        rmse, me, r_sq = results
                    if ret and rmse > 8:
                        optuna.TrialPruned()
                    rmse_list.append(rmse)
                    me_list.append(me)
                    r_sq_list.append(r_sq)
                    # wandb.log({"rmse": rmse, "me": me, "time": time}, step=pred_year)
                print("-----------")

        # save results to a csv file
        data = {
            "year": years_list,
            "run_number": run_numbers,
            "time_idx": times_list,
            "RMSE": rmse_list,
            "ME": me_list,
            "R_SQ": r_sq_list,
        }

        mean_r_sq = 0
        mean_rmse = 0
        for i in range(len(rmse_list)):
            mean_rmse += rmse_list[i]
            mean_r_sq += r_sq_list[i]
        mean_rmse = mean_rmse / len(rmse_list)
        mean_r_sq = mean_r_sq / len(r_sq_list)
        wandb.log({"mean_rmse": mean_rmse})
        wandb.log({"mean_r2": mean_r_sq})

        if self.gp is not None:
            data["RMSE_GP"] = rmse_gp_list
            data["ME_GP"] = me_gp_list
            data["R_SQ_GP"] = r_sq_gp_list

            # Calcualte average RMSE_GP for entire timespan
            mean_rmse_gp = 0
            mean_r_sq_gp = 0
            for i in range(len(rmse_gp_list)):
                mean_rmse_gp += rmse_gp_list[i]
                mean_r_sq_gp += r_sq_gp_list[i]
            mean_rmse_gp = mean_rmse_gp / len(rmse_gp_list)
            mean_r_sq_gp = mean_r_sq_gp / len(r_sq_gp_list)
            wandb.log({"mean_rmse_gp": mean_rmse_gp})
            wandb.log({"mean_r2_gp": mean_r_sq_gp})

        results_df = pd.DataFrame(data=data)
        wandb.log({"results": results_df})
        results_df.to_csv(self.savedir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv", index=False)
        if ret:
            return mean_rmse, mean_r_sq

    def _run_1_year(
            self,
            images,
            yields,
            years,
            locations,
            indices,
            predict_year,
            time,
            run_number,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
            ret,
            kfold=False,
            dropout=None
    ):
        """
        Train one model on one year of data, and then save the model predictions.
        To be called by run().
        """
        train_data, test_data = self.prepare_arrays(
            images, yields, locations, indices, years, predict_year, time
        )

        # print(train_data)

        # reinitialize the model, since self.model may be trained multiple
        # times in one call to run()
        self.reinitialize_model(time=time, dropout=dropout)

        total_size = train_data.images.shape[0]
        # "Learning rates and stopping criteria are tuned on a held-out
        # validation set (10%)."
        val_size = total_size // 10
        train_size = total_size - val_size
        print(
            f"After split, training on {train_size} examples, "
            f"validating on {val_size} examples"
        )

        if ret:
            if not kfold:
                train_indices, test_indices = train_test_split(
                    range(total_size),
                    test_size=val_size
                )

                train_dataset = Subset(TensorDataset(train_data.images, train_data.yields), train_indices)
                val_dataset = Subset(TensorDataset(train_data.images, train_data.yields), test_indices)

                train_scores, val_scores = self._train(
                    # train_data.images,
                    # train_data.yields,
                    train_dataset,
                    val_dataset,
                    train_steps,
                    batch_size,
                    starter_learning_rate,
                    weight_decay,
                    l1_weight,
                    patience,
                    predict_year,
                    ret
                )
            else:
                kf = KFold(n_splits=10, random_state=42, shuffle=True)
                kf.get_n_splits(X=range(total_size))

                all_results = None

                for i, (train_indices, test_indices) in enumerate(kf.split(X=range(total_size))):
                    self.reinitialize_model(time=time)

                    union = set(train_indices) & set(test_indices)  # temporarily for testings purposes
                    if len(union) > 0:
                        print(union)
                        exit()

                    train_dataset = Subset(TensorDataset(train_data.images, train_data.yields), train_indices)
                    val_dataset = Subset(TensorDataset(train_data.images, train_data.yields), test_indices)

                    train_scores, val_scores = self._train(
                        # train_data.images,
                        # train_data.yields,
                        train_dataset,
                        val_dataset,
                        train_steps,
                        batch_size,
                        starter_learning_rate,
                        weight_decay,
                        l1_weight,
                        patience,
                        predict_year,
                        ret
                    )

                    pdts = pd.DataFrame(train_scores)
                    pdvs = pd.DataFrame(val_scores)
                    wandb.log({"train_loss" + str(predict_year): pdts})
                    wandb.log({"val_loss" + str(predict_year): pdvs})

                    results = self._predict_opt(train_dataset, val_dataset, batch_size)

                    model_information = {
                        "state_dict": self.model.state_dict(),
                        "val_loss": val_scores["loss"],
                        "train_loss": train_scores["loss"],
                    }
                    for key in results:
                        model_information[key] = results[key]

                    # finally, get the relevant weights for the Gaussian Process
                    model_weight = self.model.state_dict()[self.model_weight]
                    model_bias = self.model.state_dict()[self.model_bias]

                    if self.model.state_dict()[self.model_weight].device != "cpu":
                        model_weight, model_bias = model_weight.cpu(), model_bias.cpu()

                    model_information["model_weight"] = model_weight.numpy()
                    model_information["model_bias"] = model_bias.numpy()

                    if self.gp is not None:
                        print("Running Gaussian Process!")
                        gp_pred = self.gp.run(
                            model_information["train_feat"],
                            model_information["test_feat"],
                            model_information["train_loc"],
                            model_information["test_loc"],
                            model_information["train_years"],
                            model_information["test_years"],
                            model_information["train_real"],
                            model_information["model_weight"],
                            model_information["model_bias"],
                        )
                        model_information["test_pred_gp"] = gp_pred.squeeze(1)

                    filename = f'{predict_year}_{run_number}_{time}_{"gp" if (self.gp is not None) else ""}.pth.tar'
                    torch.save(model_information, self.savedir / filename)

                    results = self.analyze_results(
                        model_information["test_real"],
                        model_information["test_pred"],
                        model_information["test_pred_gp"] if self.gp is not None else None,
                    )

                    if results[0] > 8:
                        raise optuna.TrialPruned()

                    if all_results is None:
                        all_results = results
                    else:
                        all_results = tuple(map(lambda j, k: j + k, all_results, results))

                all_results = tuple(map(lambda j: j / 10, all_results))
                return all_results

        else:
            train_dataset, val_dataset = random_split(
                TensorDataset(train_data.images, train_data.yields), (train_size, val_size)
            )

            train_scores, val_scores = self._train(
                # train_data.images,
                # train_data.yields,
                train_dataset,
                val_dataset,
                train_steps,
                batch_size,
                starter_learning_rate,
                weight_decay,
                l1_weight,
                patience,
                predict_year,
                ret
            )

        pdts = pd.DataFrame(train_scores)
        pdvs = pd.DataFrame(val_scores)
        wandb.log({"train_loss" + str(predict_year): pdts})
        wandb.log({"val_loss" + str(predict_year): pdvs})

        if ret:
            results = self._predict_opt(train_dataset, val_dataset, batch_size)
        else:
            results = self._predict(*train_data, *test_data, batch_size)

        model_information = {
            "state_dict": self.model.state_dict(),
            "val_loss": val_scores["loss"],
            "train_loss": train_scores["loss"],
        }
        for key in results:
            model_information[key] = results[key]

        # finally, get the relevant weights for the Gaussian Process
        model_weight = self.model.state_dict()[self.model_weight]
        model_bias = self.model.state_dict()[self.model_bias]

        if self.model.state_dict()[self.model_weight].device != "cpu":
            model_weight, model_bias = model_weight.cpu(), model_bias.cpu()

        model_information["model_weight"] = model_weight.numpy()
        model_information["model_bias"] = model_bias.numpy()

        if self.gp is not None:
            print("Running Gaussian Process!")
            gp_pred = self.gp.run(
                model_information["train_feat"],
                model_information["test_feat"],
                model_information["train_loc"],
                model_information["test_loc"],
                model_information["train_years"],
                model_information["test_years"],
                model_information["train_real"],
                model_information["model_weight"],
                model_information["model_bias"],
            )
            model_information["test_pred_gp"] = gp_pred.squeeze(1)

        filename = f'{predict_year}_{run_number}_{time}_{"gp" if (self.gp is not None) else ""}.pth.tar'
        torch.save(model_information, self.savedir / filename)
        return self.analyze_results(
            model_information["test_real"],
            model_information["test_pred"],
            model_information["test_pred_gp"] if self.gp is not None else None,
        )

    def _train(
            self,
            # train_images,
            # train_yields,
            train_dataset,  # added
            val_dataset,    # added
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
            year,
            ret,
    ):
        """Defines the training loop for a model"""

        # split the training dataset into a training and validation set
        # total_size = train_images.shape[0]
        # "Learning rates and stopping criteria are tuned on a held-out
        # validation set (10%)."
        # val_size = total_size // 10
        # train_size = total_size - val_size
        # print(
        #     f"After split, training on {train_size} examples, "
        #     f"validating on {val_size} examples"
        # )
        # train_dataset, val_dataset = random_split(
        #     TensorDataset(train_images, train_yields), (train_size, val_size)
        # )

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.Adam(
            [pam for pam in self.model.parameters()],
            lr=starter_learning_rate,
            weight_decay=weight_decay,
        )

        num_epochs = int(train_steps / (len(list(train_dataset)) / batch_size))
        print(f"Training for {num_epochs} epochs")
        wandb.config = {"total_epochs": num_epochs}

        train_scores = defaultdict(list)
        val_scores = defaultdict(list)

        step_number = 0
        min_loss = np.inf
        best_state = self.model.state_dict()

        if patience is not None:
            epochs_without_improvement = 0

        wandb.watch(self.model, log_freq=100, log_graph=True)

        for epoch in range(num_epochs):
            self.model.train()

            # running train and val scores are only for printing out
            # information
            running_train_scores = defaultdict(list)

            for train_x, train_y in tqdm(train_dataloader):
                optimizer.zero_grad()
                pred_y = self.model(train_x)

                loss, running_train_scores = l1_l2_loss(
                    pred_y, train_y, l1_weight, running_train_scores
                )
                loss.backward()
                optimizer.step()

                wandb.log({"t_loss_" + str(year): loss.item()}, step=step_number)

                train_scores["loss"].append(loss.item())

                step_number += 1

                if step_number in [4000, 20000]:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] /= 10

            train_output_strings = []
            for key, val in running_train_scores.items():
                train_output_strings.append(
                    "{}: {}".format(key, round(np.array(val).mean(), 5))
                )
                # wandb_train_loss = np.array(val).mean()

            running_val_scores = defaultdict(list)
            self.model.eval()
            with torch.no_grad():
                for (
                        val_x,
                        val_y,
                ) in tqdm(val_dataloader):
                    val_pred_y = self.model(val_x)

                    val_loss, running_val_scores = l1_l2_loss(
                        val_pred_y, val_y, l1_weight, running_val_scores
                    )

                    wandb.log({"v_loss_" + str(year): val_loss.item()}, step=epoch)
                    val_scores["loss"].append(val_loss.item())

            val_output_strings = []
            for key, val in running_val_scores.items():
                val_output_strings.append(
                    "{}: {}".format(key, round(np.array(val).mean(), 5))
                )
                # wandb_val_loss = np.array(val).mean()

            print("TRAINING: {}".format(", ".join(train_output_strings)))
            print("VALIDATION: {}".format(", ".join(val_output_strings)))

            # str_val = str(pred_year)+"_"+"val_loss"
            # str_train = str(pred_year)+"_"+"train_loss"
            # ModelBase.total_epoch += 1
            # wandb.log({"train_loss": wandb_train_loss}, step=ModelBase.total_epoch)

            epoch_val_loss = np.array(running_val_scores["loss"]).mean()

            # wandb.log({"val_loss": epoch_val_loss}, step=ModelBase.total_epoch)

            if epoch_val_loss < min_loss:
                best_state = self.model.state_dict()
                min_loss = epoch_val_loss

                if patience is not None:
                    epochs_without_improvement = 0
            elif patience is not None:
                epochs_without_improvement += 1

                if epochs_without_improvement == patience:
                    # revert to the best state dict
                    self.model.load_state_dict(best_state)
                    print("Early stopping!")
                    break

        self.model.load_state_dict(best_state)

        return train_scores, val_scores

    def _predict(
            self,
            train_images,
            train_yields,
            train_locations,
            train_indices,
            train_years,
            test_images,
            test_yields,
            test_locations,
            test_indices,
            test_years,
            batch_size,
    ):
        """
        Predict on the training and validation data. Optionally, return the last
        feature vector of the model.
        """
        train_dataset = TensorDataset(
            train_images, train_yields, train_locations, train_indices, train_years
        )

        test_dataset = TensorDataset(
            test_images, test_yields, test_locations, test_indices, test_years
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        results = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for train_im, train_yield, train_loc, train_idx, train_year in tqdm(
                    train_dataloader
            ):
                model_output = self.model(
                    train_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["train_feat"].append(feat.numpy())
                else:
                    pred = model_output
                results["train_pred"].extend(pred.squeeze(1).tolist())
                results["train_real"].extend(train_yield.squeeze(1).tolist())
                results["train_loc"].append(train_loc.numpy())
                results["train_indices"].append(train_idx.numpy())
                results["train_years"].extend(train_year.tolist())

            for test_im, test_yield, test_loc, test_idx, test_year in tqdm(
                    test_dataloader
            ):
                model_output = self.model(
                    test_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["test_feat"].append(feat.numpy())
                else:
                    pred = model_output
                results["test_pred"].extend(pred.squeeze(1).tolist())
                results["test_real"].extend(test_yield.squeeze(1).tolist())
                results["test_loc"].append(test_loc.numpy())
                results["test_indices"].append(test_idx.numpy())
                results["test_years"].extend(test_year.tolist())

        for key in results:
            if key in [
                "train_feat",
                "test_feat",
                "train_loc",
                "test_loc",
                "train_indices",
                "test_indices",
            ]:
                results[key] = np.concatenate(results[key], axis=0)
            else:
                results[key] = np.array(results[key])
        return results

    def _predict_opt(
            self,
            train_dataset,
            test_dataset,
            batch_size,
    ):
        """
        Changed copy of _predict for Optuna. Datasets from Training (train_dataset, val_dataset) are used for Validation.
        Training and Validation depending on another. Here in use fpr Hyperparameter search.
        """
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        results = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for train_im, train_yield in tqdm(
                    train_dataloader
            ):
                model_output = self.model(
                    train_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["train_feat"].append(feat.numpy())
                else:
                    pred = model_output
                results["train_pred"].extend(pred.squeeze(1).tolist())
                results["train_real"].extend(train_yield.squeeze(1).tolist())

            for test_im, test_yield in tqdm(
                    test_dataloader
            ):
                model_output = self.model(
                    test_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["test_feat"].append(feat.numpy())
                else:
                    pred = model_output
                results["test_pred"].extend(pred.squeeze(1).tolist())
                results["test_real"].extend(test_yield.squeeze(1).tolist())

        for key in results:
            if key in [
                "train_feat",
                "test_feat",
            ]:
                results[key] = np.concatenate(results[key], axis=0)
            else:
                results[key] = np.array(results[key])
        return results

    def _predict_opt2(
            self,
            train_data,
            train_ind,
            test_ind,
            batch_size,
            # results=None
    ):
        """
        Predict on the training and validation data. Optionally, return the last
        feature vector of the model.
        """
        train_images = train_data.images[train_ind]
        train_yields = train_data.yields[train_ind]
        train_locations = train_data.locations[train_ind]
        train_indices = train_data.indices[train_ind]
        train_years = train_data.years[train_ind]

        train_dataset = TensorDataset(
            train_images, train_yields, train_locations, train_indices, train_years
        )

        test_images = train_data.images[test_ind]
        test_yields = train_data.yields[test_ind]
        test_locations = train_data.locations[test_ind]
        test_indices = train_data.indices[test_ind]
        test_years = train_data.years[test_ind]

        test_dataset = TensorDataset(
            test_images, test_yields, test_locations, test_indices, test_years
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        results = defaultdict(list)
        # if results is None:
        #     results = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for train_im, train_yield, train_loc, train_idx, train_year in tqdm(
                    train_dataloader
            ):
                model_output = self.model(
                    train_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, _feature_map, feat, _layer_outputs_source = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["train_feat"].append(feat.numpy())
                else:
                    pred, _feature_map, _layer_outputs_source = model_output
                results["train_pred"].extend(pred.squeeze(1).tolist())
                results["train_real"].extend(train_yield.squeeze(1).tolist())
                results["train_loc"].append(train_loc.numpy())
                results["train_indices"].append(train_idx.numpy())
                results["train_years"].extend(train_year.tolist())

            for test_im, test_yield, test_loc, test_idx, test_year in tqdm(
                    test_dataloader
            ):
                model_output = self.model(
                    test_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, _feature_map, feat, _layer_output_target = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["test_feat"].append(feat.numpy())
                else:
                    pred, _feature_map, _layer_output_target = model_output
                results["test_pred"].extend(pred.squeeze(1).tolist())
                results["test_real"].extend(test_yield.squeeze(1).tolist())
                results["test_loc"].append(test_loc.numpy())
                results["test_indices"].append(test_idx.numpy())
                results["test_years"].extend(test_year.tolist())

        for key in results:
            if key in [
                "train_feat",
                "test_feat",
                "train_loc",
                "test_loc",
                "train_indices",
                "test_indices",
            ]:
                results[key] = np.concatenate(results[key], axis=0)
            else:
                results[key] = np.array(results[key])
        return results

    def prepare_arrays(
            self, images, yields, locations, indices, years, predict_year, time
    ):
        """Prepares the inputs for the model, in the following way:
        - normalizes the images
        - splits into a train and val set
        - turns the numpy arrays into tensors
        - removes excess months, if monthly predictions are being made
        """
        train_idx = np.nonzero(years < predict_year)[0]
        # train_idx = np.nonzero((predict_year > years) & (predict_year - 4 < years))[0]
        test_idx = np.nonzero(years == predict_year)[0]

        train_images, test_images = self._normalize(images[train_idx], images[test_idx])

        print(
            f"Train set size: {train_idx.shape[0]}, Test set size: {test_idx.shape[0]}"
        )

        Data = namedtuple("Data", ["images", "yields", "locations", "indices", "years"])

        train_data = Data(
            images=torch.as_tensor(
                train_images[:, :, :time, :], device=self.device
            ).float(),
            yields=torch.as_tensor(yields[train_idx], device=self.device)
            .float()
            .unsqueeze(1),
            locations=torch.as_tensor(locations[train_idx]),
            indices=torch.as_tensor(indices[train_idx]),
            years=torch.as_tensor(years[train_idx]),
        )

        test_data = Data(
            images=torch.as_tensor(
                test_images[:, :, :time, :], device=self.device
            ).float(),
            yields=torch.as_tensor(yields[test_idx], device=self.device)
            .float()
            .unsqueeze(1),
            locations=torch.as_tensor(locations[test_idx]),
            indices=torch.as_tensor(indices[test_idx]),
            years=torch.as_tensor(years[test_idx]),
        )

        return train_data, test_data

    @staticmethod
    def _normalize(train_images, val_images):
        """
        Find the mean values of the bands in the train images. Use these values
        to normalize both the training and validation images.
        A little awkward, since transpositions are necessary to make array broadcasting work
        """
        mean = np.mean(train_images, axis=(0, 2, 3))

        train_images = (train_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)
        val_images = (val_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)

        return train_images, val_images

    @staticmethod
    def analyze_results(true, pred, pred_gp):
        """Calculate ME and RMSE"""
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        me = np.mean(true - pred)
        r_sq = r2_score(true, pred)

        fig, ax = plt.subplots()
        ax.scatter(true, pred)
        ax.annotate("r-squared = {:.3f}".format(r2_score(true, pred)), (0, 1))
        ax.annotate("rmse = {:.3f}".format(rmse), (3, 0))
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        y_test, y_predicted = true.reshape(-1, 1), pred.reshape(-1, 1)
        x = np.linspace(0, 60, 1)
        ax.plot(y_test, LinearRegression().fit(y_test, y_predicted).predict(y_test), '-r')
        ax.plot(x, x)

        print(f"Without GP: RMSE: {rmse}, ME: {me}, r2: {r_sq}")

        if pred_gp is not None:
            rmse_gp = np.sqrt(np.mean((true - pred_gp) ** 2))
            me_gp = np.mean(true - pred_gp)
            r_sq_gp = r2_score(true, pred_gp)
            print(f"With GP: RMSE: {rmse_gp}, ME: {me_gp}, r2: {r_sq_gp}")
            return rmse, me, rmse_gp, me_gp, r_sq, r_sq_gp
        return rmse, me, r_sq

    def reinitialize_model(self, time=None, dropout=None):
        raise NotImplementedError
