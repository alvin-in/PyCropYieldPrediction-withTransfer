import pandas as pd
import torch
from pathlib import Path

from cyp.data import MODISExporter, DataCleaner, Engineer, ArgDataCleaner, Arg_Engineer, YieldDataCleansing
from cyp.models import ConvModel, RNNModel, TransConvModel
from cyp.analysis import counties_plot, vis_argentina

import fire
import wandb
import optuna
import matplotlib.pyplot as plt
import joblib


# from torchsummary import summary
# from cyp.models.base import ModelBase


class RunTask:
    """Entry point into the pipeline.

    For convenience, all the parameter descriptions are copied from the classes.
    """

    @staticmethod
    def export(
            export_limit=None,
            major_states_only=False,
            check_if_done=True,
            download_folder=None,
            yield_data_path="data/yield_data.csv",
    ):
        """
        Export all the data necessary to train the models.

        Parameters
        ----------
        export_limit: int or None, default=None
            If not none, how many .tif files to export (*3, for the image, mask and temperature
            files)
        major_states_only: boolean, default=True
            Whether to only use the 11 states responsible for 75 % of national soybean
            production, as is done in the paper
        check_if_done: boolean, default=False
            If true, will check download_folder for any .tif files which have already been
            downloaded, and won't export them again. This effectively allows for
            checkpointing, and prevents all files from having to be downloaded at once.
        download_folder: None or pathlib Path, default=None
            Which folder to check for downloaded files, if check_if_done=True. If None, looks
            in data/folder_name
        yield_data_path: str, default='data/yield_data.csv'
            A path to the yield data
        """
        yield_data_path = Path(yield_data_path)
        exporter = MODISExporter(locations_filepath=yield_data_path)
        exporter.export_all(
            export_limit, major_states_only, check_if_done, download_folder
        )

    @staticmethod
    def process(
            mask_path="data/crop_yield-data_mask",
            temperature_path="data/crop_yield-data_temperature",
            image_path="data/crop_yield-data_image",
            yield_data_path="data/usa_yield_with_pix2.csv",
            cleaned_data_path="data/img_output2020",
            multiprocessing=False,
            processes=4,
            parallelism=6,
            delete_when_done=False,
            num_years=11,
            checkpoint=True,
    ):
        """
        Preprocess the data

        Parameters
        ----------
        mask_path: str, default='data/crop_yield-data_mask'
            Path to which the mask tif files have been saved
        temperature_path: str, default='data/crop_yield-data_temperature'
            Path to which the temperature tif files have been saved
        image_path: str, default='data/crop_yield-data_image'
            Path to which the image tif files have been saved
        yield_data_path: str, default='data/yield_data.csv'
            Path to the yield data csv file
        cleaned_data_path: str, default='data/img_output'
            Path to save the data to
        multiprocessing: boolean, default=False
            Whether to use multiprocessing
        processes: int, default=4
            Number of processes to use if multiprocessing=True
        parallelism: int, default=6
            Parallelism if multiprocesisng=True
        delete_when_done: boolean, default=False
            Whether or not to delete the original .tif files once the .npy array
            has been generated.
        num_years: int, default=14
            How many years of data to create.
        checkpoint: boolean, default=True
            Whether or not to skip tif files which have already had their .npy arrays
            written
        """
        mask_path = Path(mask_path)
        temperature_path = Path(temperature_path)
        image_path = Path(image_path)
        yield_data_path = Path(yield_data_path)
        cleaned_data_path = Path(cleaned_data_path)

        cleaner = DataCleaner(
            mask_path,
            temperature_path,
            image_path,
            yield_data_path,
            savedir=cleaned_data_path,
            multiprocessing=multiprocessing,
            processes=processes,
            parallelism=parallelism,
        )
        cleaner.process(
            delete_when_done=delete_when_done,
            num_years=num_years,
            checkpoint=checkpoint,
        )

    @staticmethod
    def engineer(
            cleaned_data_path="data/img_output2020",
            yield_data_path="data/usa_yield_with_pix2.csv",
            county_data_path="data/county_data.csv",
            num_bins=13,
            max_bin_val=4999,
    ):
        """
        Take the preprocessed data and generate the input to the models

        Parameters
        ----------
        cleaned_data_path: str, default='data/img_output'
            Path to save the data to, and path to which processed data has been saved
        yield_data_path: str, default='data/yield_data.csv'
            Path to the yield data csv file
        county_data_path: str, default='data/county_data.csv'
            Path to the county data csv file
        num_bins: int, default=32
            If generate=='histogram', the number of bins to generate in the histogram.
        max_bin_val: int, default=4999
            The maximum value of the bins. The default is taken from the original paper;
            note that the maximum pixel values from the MODIS datsets range from 16000 to
            18000 depending on the band
        """
        cleaned_data_path = Path(cleaned_data_path)
        yield_data_path = Path(yield_data_path)
        county_data_path = Path(county_data_path)

        engineer = Engineer(cleaned_data_path, yield_data_path, county_data_path)
        engineer.process(
            num_bands=9,
            generate="histogram",
            num_bins=num_bins,
            max_bin_val=max_bin_val,
            channels_first=True,
        )

    @staticmethod
    def train_cnn(
            cleaned_data_path=Path("data/img_output2020"),  # img_output2000"),
            dropout=0.396786878766754,    # 0.43356673510426585,    # 0.324,  # 0.295111873921675,
            dense_features=None,
            savedir=Path("data/us_with2020_checkup2"),
            times="all",  # "all",
            pred_years=range(2015, 2021),
            num_runs=4,
            train_steps=27000,  # 46000,  # 22050,  # 23000,
            batch_size=32,
            starter_learning_rate=0.00576923160031681,  # 0.008,    # 0.009955851899640822,  # 0.00690285122948098,
            weight_decay=0,
            l1_weight=0,
            patience=16,   # 29,   # 10,  # 44,
            use_gp=True,
            sigma=1,
            r_loc=0.5,
            r_year=1.5,
            sigma_e=0.32,
            sigma_b=0.01,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            multi_hyp=None  # Path("data/out_hyp_us.csv")
    ):
        """
        Train a CNN model

        Parameters
        ----------
        cleaned_data_path: str, default='data/img_output'
            Path to which histogram has been saved
        dropout: float, default=0.5
            Default taken from the original paper
        dense_features: list, or None, default=None.
            output feature size of the Linear layers. If None, default values will be taken from the paper.
            The length of the list defines how many linear layers are used.
        savedir: pathlib Path, default=Path('data/models')
            The directory into which the models should be saved.
        times: {'all', 'realtime'}
            Which time indices to train the model on. If 'all', a full run (32 timesteps) is used.
            If 'realtime', range(10, 31, 4) is used.
        pred_years: int or None, default=None
            Which year to build models for. If None, the default values from the paper (range(2009, 2016))
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
            In addition to MSE, L1 loss is also used (sometimes). The default is 0, but a value of 1.5 is used
            when training the model in batch
        patience: int or None, default=10
            The number of epochs to wait without improvement in the validation loss before terminating training.
            Note that the original repository doesn't use early stopping.

        use_gp: boolean, default=True
            Whether to use a Gaussian process in addition to the model

        If use_gp=True, the following parameters are also used:

        sigma: float, default=1
            The kernel variance, or the signal variance
        r_loc: float, default=0.5
            The length scale for the location data (latitudes and longitudes)
        r_year: float, default=1.5
            The length scale for the time data (years)
        sigma_e: float, default=0.32
            Noise variance. 0.32 **2 ~= 0.1
        sigma_b: float, default=0.01
            Parameter variance; the variance on B

        device: torch.device
            Device to run model on. By default, checks for a GPU. If none exists, uses
            the CPU
        multi_hyp: Path, default=None
            If not none: Path to csv with yearly hyperparameters.
        """
        wandb.init(project='crop_yield_prediction_7', config={"train_steps": train_steps,
                                                              "batch_size": batch_size,
                                                              "starter_learning_rate": starter_learning_rate,
                                                              "dropout": dropout,
                                                              "use_gp": use_gp,
                                                              "pred_years": pred_years,
                                                              })

        histogram_path = Path(cleaned_data_path) / "histogram_all_full.npz"

        model = ConvModel(
            in_channels=9,
            dropout=dropout,
            dense_features=dense_features,
            savedir=savedir,
            use_gp=use_gp,
            sigma=sigma,
            r_loc=r_loc,
            r_year=r_year,
            sigma_e=sigma_e,
            sigma_b=sigma_b,
            device=device,
        )

        model.run(
            histogram_path,
            times,
            pred_years,
            num_runs,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
            multi_hyp=multi_hyp
        )

    # in Bearbeitung
    @staticmethod
    def train_trans_cnn(
            cleaned_data_path=Path("data/arg_im_out_with2020"),
            second_hist_path=None,  # Path("data/img_output/histogram_all_full.npz"),
            dropout=0.5,    # 0.5,
            dense_features=None,
            savedir=Path("data/arg_us_init_freezing"),
            times="all",
            pred_years=range(2018, 2021),
            num_runs=4,
            train_steps=25000,
            batch_size=32,
            starter_learning_rate=0.000109716901096571,  # 0.001,     # 1e-4,
            weight_decay=1,
            l1_weight=0,
            patience=10,  # 20
            use_gp=True,
            sigma=1,  # 0.823,    # 1.1358229629150707,   # 1,
            r_loc=0.5,  # 0.386,      # 0.36160527453661073,  # 0.5,
            r_year=1.5,  # 1.584,     # 1.9723720455649654,  # 1.5,
            sigma_e=0.32,  # 0.025,    # 0.4256803798519013,     # 0.32,
            sigma_b=0.1,  # 0.02,    # 0.019183576228164204,   # 0.1,
            nu=1.5,  # 1.5,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            freeze=4,  # 4,
            us_init=True,  # True,
            sp_weight=0,    # 0.2292288847991445,    # 0,
            l2_sp_beta=0,   # 0.7716633924322595,   # 0,
            bss=0,  # 0.0714846512298477,  # 0.412,
            bss_k=0,
            delta=0,
            delta_w_att=0,
            multi_hyp=None  # Path("data/out_hyp.csv")
    ):
        wandb.init(project='crop_yield_prediction_8', config={"type": "Transfer_Learning",
                                                              "train_steps": train_steps,
                                                              "batch_size": batch_size,
                                                              "starter_learning_rate": starter_learning_rate,
                                                              "dropout": dropout,
                                                              "use_gp": use_gp,
                                                              "pred_years": pred_years,
                                                              "l1": l1_weight,
                                                              "weight_decay": weight_decay,
                                                              "freeze": freeze,
                                                              "us_init": us_init,
                                                              "sp_weight": sp_weight,
                                                              "l2_sp_beta": l2_sp_beta,
                                                              "bss": bss,
                                                              "bss_k": bss_k,
                                                              "delta": delta,
                                                              "delta_w_att": delta_w_att,
                                                              "nu": nu,
                                                              })

        histogram_path = Path(cleaned_data_path) / "histogram_all_full.npz"

        model = TransConvModel(
            in_channels=9,
            dropout=dropout,
            dense_features=dense_features,
            savedir=savedir,
            use_gp=use_gp,
            sigma=sigma,
            r_loc=r_loc,
            r_year=r_year,
            sigma_e=sigma_e,
            sigma_b=sigma_b,
            nu=nu,
            device=device,
        )

        # model.model.load_state_dict(
        #    torch.load(Path("H:\\BA\\pycrop-yield-prediction\\data\\models\\cnn\\2015_2_32_gp.pth.tar"))["state_dict"])
        learn_rate_decay = [3364, 19678]  # [368, 11050]# [2000, 20000]   # 20000]

        model.run(
            histogram_path,
            second_hist_path,
            times,
            pred_years,
            num_runs,
            train_steps=train_steps,
            batch_size=batch_size,
            starter_learning_rate=starter_learning_rate,
            learn_rate_decay=learn_rate_decay,
            weight_decay=weight_decay,
            l1_weight=l1_weight,
            patience=patience,
            freeze=freeze,
            us_init=us_init,
            sp_weight=sp_weight,
            l2_sp_beta=l2_sp_beta,
            bss=bss,
            bss_k=bss_k,
            delta=delta,
            delta_w_att=delta_w_att,
            multi_hyp=multi_hyp
        )

        vis_argentina.vis_arg(pred_years)

    @staticmethod
    def hyp_multi_trans_cnn(
            cleaned_data_path=Path("data/arg_im_out_with2020"),
            hyp_path=Path("data/hyp"),
            second_hist_path=None,  # Path("data/img_output/histogram_all_full.npz"),
            dropout=0.5,
            dense_features=None,
            savedir=Path("data/arg_models_32steps_5_yet"),
            times="all",
            pred_years=range(2015, 2021),
            num_runs=4,
            train_steps=25000,
            batch_size=32,
            starter_learning_rate=1e-4,  # 0.001,     # 1e-4,
            weight_decay=1,  # 0,
            l1_weight=0,
            patience=10,  # 20
            use_gp=True,
            sigma=1,  # 0.823,    # 1.1358229629150707,   # 1,
            r_loc=0.5,  # 0.386,      # 0.36160527453661073,  # 0.5,
            r_year=1.5,  # 1.584,     # 1.9723720455649654,  # 1.5,
            sigma_e=0.32,  # 0.025,    # 0.4256803798519013,     # 0.32,
            sigma_b=0.1,  # 0.02,    # 0.019183576228164204,   # 0.1,
            nu=1.5,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            freeze=4,  # 4,
            us_init=True,  # True,
            sp_weight=0,
            l2_sp_beta=0,
            bss=0.412,
            bss_k=1,
            delta=0,
            delta_w_att=0,
            out_hyp_csv="H:\\BA\\pycrop-yield-prediction\\data\\out_hyp.csv"
    ):
        wandb.init(project='crop_yield_prediction_4', config={"type": "Transfer_Learning; mult Hyp",
                                                              "train_steps": train_steps,
                                                              "batch_size": batch_size,
                                                              "starter_learning_rate": starter_learning_rate,
                                                              "dropout": dropout,
                                                              "use_gp": use_gp,
                                                              "pred_years": pred_years,
                                                              "l1": l1_weight,
                                                              "weight_decay": weight_decay,
                                                              "freeze": freeze,
                                                              "us_init": us_init,
                                                              "sp_weight": sp_weight,
                                                              "l2_sp_beta": l2_sp_beta,
                                                              "bss": bss,
                                                              "bss_k": bss_k,
                                                              "delta": delta,
                                                              "delta_w_att": delta_w_att,
                                                              "nu": nu
                                                              })

        histogram_path = Path(cleaned_data_path) / "histogram_all_full.npz"

        # df = pd.DataFrame(columns=['year', 'learn_rate_decay1', 'learn_rate_decay2', 'train_steps', 'weight_decay',
        #                            'l1_weight', 'sp_weight', 'l2_sp_beta', 'bss', 'bss_k'])
        df = None

        wandb.init(project='crop_yield_prediction_4', config={"type": "Optuna-Optimization",
                                                              "train_steps": train_steps,
                                                              "batch_size": batch_size,
                                                              "starter_learning_rate": starter_learning_rate,
                                                              "dropout": dropout,
                                                              "use_gp": use_gp,
                                                              "pred_years": pred_years,
                                                              "freeze": freeze,
                                                              })

        histogram_path = Path(cleaned_data_path) / "histogram_all_full.npz"

        for p_year in pred_years:
            study = optuna.create_study(direction="minimize")  # , "maximize"])

            def objective(trial):
                model = TransConvModel(
                    in_channels=9,
                    dropout=0.5,  # trial.suggest_float("dropout", 0.2, 0.6),
                    dense_features=dense_features,
                    savedir=savedir,
                    use_gp=None,
                    sigma=sigma,    # trial.suggest_float("sigma", 0.5, 1.5),  # sigma,
                    r_loc=r_loc,    # trial.suggest_float("r_loc", 0.3, 0.7),  # r_loc,
                    r_year=r_year,  # trial.suggest_float("r_year", 1, 2),  # r_year,
                    sigma_e=sigma_e,    # trial.suggest_float("sigma_e", 0, 1),  # sigma_e,
                    sigma_b=sigma_b,    # trial.suggest_float("sigma_b", 0, 0.1),  # sigma_b,
                    nu=nu,  # trial.suggest_float("nu", 0.5, 15, step=0.5),
                    device=device,
                )

                """
                sigma=trial.suggest_float("sigma", 0.7, 1.3),   # sigma,
                    r_loc=trial.suggest_float("r_loc", 0.3, 0.7),   # r_loc,
                    r_year=trial.suggest_float("r_year", 1, 2),    # r_year,
                    sigma_e=trial.suggest_float("sigma_e", 0.1, 5),   # sigma_e,
                    sigma_b=trial.suggest_float("sigma_b", 0, 0.1),   # sigma_b,
                """

                learn_rate_decay = [trial.suggest_int("learn_rate_decay1", 200, 3000),
                                    trial.suggest_int("learn_rate_decay2", 3000, 20000)]

                rmse, rsq = model.run(
                    histogram_path,
                    second_hist_path,
                    times,
                    pred_years=[p_year],
                    num_runs=1,
                    train_steps=trial.suggest_int("train_steps", 5000, 30000, 1000),
                    batch_size=32,
                    starter_learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2),
                    learn_rate_decay=learn_rate_decay,
                    weight_decay=trial.suggest_float("weighty_decay", 0, 1),
                    l1_weight=0,  # trial.suggest_int("l1_weights", 0, 1),
                    patience=10,  # trial.suggest_int("patience", 0, 50),
                    freeze=trial.suggest_int("freeze", 3, 5),
                    us_init=True,  # trial.suggest_int("us-init", 0, 1),
                    ret=True,
                    sp_weight=trial.suggest_float("sp_weight", 0, 1),
                    l2_sp_beta=trial.suggest_float("l2_sp_beta", 0, 1),
                    bss=trial.suggest_float("bss", 0, 1),  # 0.412,  # trial.suggest_float("bss", 0, 1),
                    bss_k=1,  # 1     # trial.suggest_int("bss_k", 0, 32)
                    kfold=True
                )

                return rmse  # , rsq

            study.optimize(objective, n_trials=30)
            print(study.best_trial.params)
            # trial_w_best_rmse = max(study.best_trials, key=lambda t: t.values[0])
            p_df = pd.DataFrame(study.best_trial.params, index=[0])
            p_df['year'] = p_year
            if df is None:
                df = p_df
            else:
                df = pd.concat([df, p_df])

        df.to_csv(out_hyp_csv)

    @staticmethod
    def hyp_multi_trans_cnn_usa(
            cleaned_data_path=Path("I:/US-SAT-DS/img_output2020"),
            dropout=0.5,
            dense_features=None,
            savedir=Path("data/us_models_hyp"),
            times="all",
            pred_years=range(2015, 2021),
            num_runs=1,
            train_steps=25000,
            batch_size=32,
            starter_learning_rate=1e-3,
            weight_decay=1,
            l1_weight=0,
            patience=10,
            use_gp=None,
            sigma=1,
            r_loc=0.5,
            r_year=1.5,
            sigma_e=0.32,
            sigma_b=0.01,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            out_hyp_csv="H:\\BA\\pycrop-yield-prediction\\data\\out_hyp_us.csv"
    ):
        wandb.init(project='crop_yield_prediction_4', config={"type": "mult Hyp",
                                                              "train_steps": train_steps,
                                                              "batch_size": batch_size,
                                                              "starter_learning_rate": starter_learning_rate,
                                                              "dropout": dropout,
                                                              "use_gp": use_gp,
                                                              "pred_years": pred_years,
                                                              "l1": l1_weight,
                                                              "weight_decay": weight_decay
                                                              })

        histogram_path = Path(cleaned_data_path) / "histogram_all_full.npz"

        # df = pd.DataFrame(columns=['year', 'learn_rate_decay1', 'learn_rate_decay2', 'train_steps', 'weight_decay',
        #                            'l1_weight', 'sp_weight', 'l2_sp_beta', 'bss', 'bss_k'])
        df = None

        wandb.init(project='crop_yield_prediction_4', config={"type": "Optuna-Optimization USA multi",
                                                              "train_steps": train_steps,
                                                              "batch_size": batch_size,
                                                              "starter_learning_rate": starter_learning_rate,
                                                              "dropout": dropout,
                                                              "use_gp": use_gp,
                                                              "pred_years": pred_years
                                                              })

        histogram_path = Path(cleaned_data_path) / "histogram_all_full.npz"

        for p_year in pred_years:
            study = optuna.create_study(direction="minimize")  # , "maximize"])

            def objective(trial):
                joblib.dump(study, "data/optuna_study_ave_new.pkl")

                model = ConvModel(
                    in_channels=9,
                    dropout=trial.suggest_float("dropout", 0, 1),
                    dense_features=dense_features,
                    savedir=savedir,
                    use_gp=None,
                    sigma=sigma,
                    r_loc=r_loc,
                    r_year=r_year,
                    sigma_e=sigma_e,
                    sigma_b=sigma_b,
                    device=device,
                )

                rmse, _rsq = model.run(
                    histogram_path,
                    times,
                    p_year,
                    num_runs=1,
                    train_steps=trial.suggest_int("train_steps", 5000, 50000, 1000),
                    batch_size=32,
                    starter_learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2),
                    weight_decay=trial.suggest_int("weighty_decay", 0, 1),
                    l1_weight=0,  # trial.suggest_int("l1_weight", 0, 1),
                    patience=10,    # trial.suggest_int("patience", 0, 40),
                    ret=True,
                    kfold=True
                )

                study.trials_dataframe().to_csv("data/optuna_results_usa3.csv")

                return rmse

            study.optimize(objective, n_trials=30)
            # trial_w_best_rmse = max(study.best_trials, key=lambda t: t.values[0])
            p_df = pd.DataFrame(study.best_trial.params, index=[0])
            p_df['year'] = p_year
            if df is None:
                df = p_df
            else:
                df = pd.concat([df, p_df])

        df.to_csv(out_hyp_csv)

    @staticmethod
    def train_rnn(
            cleaned_data_path="data/img_output",
            num_bins=32,
            hidden_size=128,
            rnn_dropout=0.75,
            dense_features=None,
            savedir=Path("data/models"),
            times="all",
            pred_years=None,
            num_runs=2,
            train_steps=10000,
            batch_size=32,
            starter_learning_rate=1e-3,
            weight_decay=0,
            l1_weight=0,
            patience=10,
            use_gp=True,
            sigma=1,
            r_loc=0.5,
            r_year=1.5,
            sigma_e=0.32,
            sigma_b=0.01,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Train an RNN model

        Parameters
        ----------
        cleaned_data_path: str, default='data/img_output'
            Path to which histogram has been saved
        num_bins: int, default=32
            Number of bins in the generated histogram
        hidden_size: int, default=128
            The size of the hidden state. Default taken from the original paper
        rnn_dropout: float, default=0.75
            Default taken from the original paper. Note that this dropout is applied to the
            hidden state after each timestep, not after each layer (since there is only one layer)
        dense_features: list, or None, default=None.
            output feature size of the Linear layers. If None, default values will be taken from the paper.
            The length of the list defines how many linear layers are used.
        savedir: pathlib Path, default=Path('data/models')
            The directory into which the models should be saved.
        times: {'all', 'realtime'}
            Which time indices to train the model on. If 'all', a full run (32 timesteps) is used.
            If 'realtime', range(10, 31, 4) is used.
        pred_years: int or None, default=None
            Which years to build models for. If None, the default values from the paper (range(2009, 2016))
            are used.
        num_runs: int, default=2
            The number of runs to do per year. Default taken from the paper
        train_steps: int, default=10000
            The number of steps for which to train the model. Default taken from the paper.
        batch_size: int, default=32
            Batch size when training. Default taken from the paper
        starter_learning_rate: float, default=1e-3
            Starter learning rate. Note that the learning rate is divided by 10 after 2000 and 4000 training
            steps. Default taken from the paper
        weight_decay: float, default=1
            Weight decay (L2 regularization) on the model weights
        l1_weight: float, default=0
            L1 loss is not used for the RNN. Setting it to 0 avoids it being computed.
        patience: int or None, default=10
            The number of epochs to wait without improvement in the validation loss before terminating training.
            Note that the original repository doesn't use early stopping.

        use_gp: boolean, default=True
            Whether to use a Gaussian process in addition to the model

        If use_gp=True, the following parameters are also used:

        sigma: float, default=1
            The kernel variance, or the signal variance
        r_loc: float, default=0.5
            The length scale for the location data (latitudes and longitudes)
        r_year: float, default=1.5
            The length scale for the time data (years)
        sigma_e: float, default=0.32
            Noise variance. 0.32 **2 ~= 0.1
        sigma_b: float, default=0.01
            Parameter variance; the variance on B

        device: torch.device
            Device to run model on. By default, checks for a GPU. If none exists, uses
            the CPU

        """
        histogram_path = Path(cleaned_data_path) / "histogram_all_full.npz"

        model = RNNModel(
            in_channels=9,
            num_bins=num_bins,
            hidden_size=hidden_size,
            rnn_dropout=rnn_dropout,
            dense_features=dense_features,
            savedir=savedir,
            use_gp=use_gp,
            sigma=sigma,
            r_loc=r_loc,
            r_year=r_year,
            sigma_e=sigma_e,
            sigma_b=sigma_b,
            device=device,
        )
        model.run(
            histogram_path,
            times,
            pred_years,
            num_runs,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
        )

    @staticmethod
    def visualization(model_path):
        """
        Generates an svg of the counties, coloured by their prediction error.
        Parameters
        ----------
        model_path: str
        Path to model (.pth.tar-file) and output into these parent directory
        """
        counties_plot.plot_county_errors(model_path)

    @staticmethod
    def process_argentina(
            mask_path="data/cover",
            temperature_path="data/temp",
            image_path="data/sat",
            yield_data_path='data/yield_data_with2020.csv',  # "data/soja-serie-1969-2019(3).csv",
            cleaned_data_path="data/arg_im_out_with2020",
            multiprocessing=False,
            processes=4,
            parallelism=6,
            delete_when_done=False,
            num_years=12,
            checkpoint=False,
    ):
        """
        Preprocess the data (Argentina). Description of params not changed!

        Parameters
        ----------
        mask_path: str, default='data/crop_yield-data_mask'
            Path to which the mask tif files have been saved
        temperature_path: str, default='data/crop_yield-data_temperature'
            Path to which the temperature tif files have been saved
        image_path: str, default='data/crop_yield-data_image'
            Path to which the image tif files have been saved
        yield_data_path: str, default='data/yield_data.csv'
            Path to the yield data csv file
        cleaned_data_path: str, default='data/img_output'
            Path to save the data to
        multiprocessing: boolean, default=False
            Whether to use multiprocessing
        processes: int, default=4
            Number of processes to use if multiprocessing=True
        parallelism: int, default=6
            Parallelism if multiprocesisng=True
        delete_when_done: boolean, default=False
            Whether or not to delete the original .tif files once the .npy array
            has been generated.
        num_years: int, default=14
            How many years of data to create.
        checkpoint: boolean, default=True
            Whether or not to skip tif files which have already had their .npy arrays
            written
        """
        mask_path = Path(mask_path)
        temperature_path = Path(temperature_path)
        image_path = Path(image_path)
        yield_data_path = Path(yield_data_path)
        cleaned_data_path = Path(cleaned_data_path)

        cleaner = ArgDataCleaner(
            mask_path,
            temperature_path,
            image_path,
            yield_data_path,
            savedir=cleaned_data_path,
            multiprocessing=multiprocessing,
            processes=processes,
            parallelism=parallelism,
        )
        cleaner.process(
            delete_when_done=delete_when_done,
            num_years=num_years,
            checkpoint=checkpoint,
        )

    @staticmethod
    def arg_engineer(
            cleaned_data_path="data/arg_im_out_with2020",  # "data/arg_im_out",
            yield_data_path='data/yield_data_with2020.csv',  # "data/soja-serie-1969-2019(3).csv",
            county_data_path="data/departamentos.csv",
            num_bins=13,
            max_bin_val=4999,
    ):
        """
        Take the preprocessed data and generate the input to the models

        Parameters
        ----------
        cleaned_data_path: str, default='data/img_output'
            Path to save the data to, and path to which processed data has been saved
        yield_data_path: str, default='data/yield_data.csv'
            Path to the yield data csv file
        county_data_path: str, default='data/county_data.csv'
            Path to the county data csv file
        num_bins: int, default=32
            If generate=='histogram', the number of bins to generate in the histogram.
        max_bin_val: int, default=4999
            The maximum value of the bins. The default is taken from the original paper;
            note that the maximum pixel values from the MODIS datsets range from 16000 to
            18000 depending on the band
        """
        cleaned_data_path = Path(cleaned_data_path)
        yield_data_path = Path(yield_data_path)
        county_data_path = Path(county_data_path)

        engineer = Arg_Engineer(cleaned_data_path, yield_data_path, county_data_path)
        engineer.process(
            num_bands=9,
            generate="histogram",
            num_bins=num_bins,
            max_bin_val=max_bin_val,
            channels_first=True,
        )

    @staticmethod
    def data_cleansing(
            mask_path="data/crop_yield-data_mask",  # data/cover for Argentina
            num_years=11,   # use +1 for Argentina
            out_name="usa_pix_counter",     # 'arg_pix_counter' for Argentina
    ):
        """
        Calculates pixel count per year for each county. Output is a csv into data and can be used to modify yield data.

        Parameters
        ----------
        mask_path: str, default='data/crop_yield-data_mask'
            Path from which the mask tif files get loaded
        num_years: int, default=11
            How many years of data to create.
        out_name: string
            Name of resulting csv.
        """
        mask_path = Path(mask_path)

        cleaner = YieldDataCleansing(mask_path)
        cleaner.process(num_years=num_years, out=out_name)

    @staticmethod
    def run_optuna(
            cleaned_data_path=Path("data/arg_im_out"),
            second_hist_path=None,  # Path("data/img_output/histogram_all_full.npz"),
            dropout=0.5,
            dense_features=None,
            savedir=Path("data/arg_hyp_clean_only_"),
            times="all",
            pred_years=range(2015, 2018),
            num_runs=4,
            train_steps=25000,
            batch_size=32,
            starter_learning_rate=1e-4,
            weight_decay=0,
            l1_weight=0,
            patience=10,
            use_gp=True,
            sigma=1,
            r_loc=0.5,
            r_year=1.5,
            sigma_e=0.32,
            sigma_b=0.01,
            nu=1.5,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            freeze=4,
            sp_weight=0,
            l2_sp_beta=0,
            bss=0,
    ):
        wandb.init(project='crop_yield_prediction_4', config={"type": "Optuna-Optimization",
                                                              "train_steps": train_steps,
                                                              "batch_size": batch_size,
                                                              "starter_learning_rate": starter_learning_rate,
                                                              "dropout": dropout,
                                                              "use_gp": use_gp,
                                                              "pred_years": pred_years,
                                                              "freeze": freeze,
                                                              })

        histogram_path = Path(cleaned_data_path) / "histogram_all_full.npz"

        study = optuna.create_study(directions=["minimize", "maximize"])

        def objective(trial):
            model = TransConvModel(
                in_channels=9,
                dropout=0.5,    # trial.suggest_float("dropout", 0.2, 0.6),
                dense_features=dense_features,
                savedir=savedir,
                use_gp=True,
                sigma=sigma,    # trial.suggest_float("sigma", 0.5, 1.5),  # sigma,
                r_loc=r_loc,    # trial.suggest_float("r_loc", 0.3, 0.7),  # r_loc,
                r_year=r_year,  # trial.suggest_float("r_year", 1, 2),  # r_year,
                sigma_e=sigma_e,    # trial.suggest_float("sigma_e", 0, 1),  # sigma_e,
                sigma_b=sigma_b,    # trial.suggest_float("sigma_b", 0, 0.1),  # sigma_b,
                nu=nu,  # trial.suggest_float("nu", 0.5, 15, step=0.5),
                device=device,
            )

            """
            sigma=trial.suggest_float("sigma", 0.7, 1.3),   # sigma,
                r_loc=trial.suggest_float("r_loc", 0.3, 0.7),   # r_loc,
                r_year=trial.suggest_float("r_year", 1, 2),    # r_year,
                sigma_e=trial.suggest_float("sigma_e", 0.1, 5),   # sigma_e,
                sigma_b=trial.suggest_float("sigma_b", 0, 0.1),   # sigma_b,
            """

            # model.model.load_state_dict(
            #    torch.load(Path("H:\\BA\\pycrop-yield-prediction\\data\\models\\cnn\\2015_2_32_gp.pth.tar"))["state_dict"])

            learn_rate_decay = [trial.suggest_int("learn_rate_decay1", 1000, 4000),
                                trial.suggest_int("learn_rate_decay2", 4000, 20000)]
            # learn_rate_decay = [2000, 20000]   # 20000]

            rmse, rsq = model.run(
                histogram_path,
                second_hist_path,
                times,
                pred_years=pred_years,
                num_runs=1,
                train_steps=25000,  # trial.suggest_int("train_steps", 5000, 30000, 1000),
                batch_size=32,
                starter_learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2),
                learn_rate_decay=learn_rate_decay,
                weight_decay=1,     # trial.suggest_float("weighty_decay", 0, 1),
                l1_weight=0,  # trial.suggest_int("l1_weights", 0, 1),
                patience=10,    # trial.suggest_int("patience", 0, 50),
                freeze=0,
                us_init=True,   # trial.suggest_int("us-init", 0, 1),
                ret=True,
                sp_weight=0,    # trial.suggest_float("sp_weight", 0, 1),
                l2_sp_beta=0,    # trial.suggest_float("l2_sp_beta", 0, 1),
                bss=0,  # trial.suggest_float("bss", 0, 1),
                bss_k=0    # trial.suggest_int("bss_k", 0, 1)
            )

            study.trials_dataframe().to_csv("data/arg_hyp_clean_all.csv")
            joblib.dump(study, "data/arg_hyp_clean_all.pkl")
            return rmse, rsq

        study.optimize(objective, n_trials=30)
        # print('Number of finished trials:', len(study.trials))
        # print('Best trial:', study.best_trial.params)
        trial_w_best_rmse = max(study.best_trials, key=lambda t: t.values[1])
        print(f"Trial with highest R^2: ")
        print(f"\tnumber: {trial_w_best_rmse.number}")
        print(f"\tparams: {trial_w_best_rmse.params}")
        print(f"\tvalues: {trial_w_best_rmse.values}")
        fig = optuna.visualization.plot_pareto_front(study, target_names=["RMSE", "R²"])
        fig.show()
        fig2 = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[0], target_name="RMSE"
        )
        fig2.show()
        # fig = optuna.visualization.plot_optimization_history(study)
        # fig.show()
        # fig2 = optuna.visualization.plot_parallel_coordinate(study)
        # fig2.show()
        # fig3 = optuna.visualization.plot_param_importances(study)
        # fig3.show()
        joblib.dump(study, "data/arg_hyp_clean_all.pkl")
        # vis_argentina.vis_arg(pred_years)

    @staticmethod
    def run_optuna_usa(
            cleaned_data_path=Path("I:/US-SAT-DS/img_output2020"),
            dropout=0.5,
            dense_features=None,
            savedir=Path("data/arg_models_"),
            times="all",
            pred_years=range(2012, 2015),
            num_runs=1,
            train_steps=25000,
            batch_size=32,
            starter_learning_rate=1e-3,
            weight_decay=1,
            l1_weight=0,
            patience=10,
            use_gp=None,
            sigma=1,
            r_loc=0.5,
            r_year=1.5,
            sigma_e=0.32,
            sigma_b=0.01,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Copy of: Train a CNN model

        Parameters
        ----------
        cleaned_data_path: str, default='data/img_output'
            Path to which histogram has been saved
        dropout: float, default=0.5
            Default taken from the original paper
        dense_features: list, or None, default=None.
            output feature size of the Linear layers. If None, default values will be taken from the paper.
            The length of the list defines how many linear layers are used.
        savedir: pathlib Path, default=Path('data/models')
            The directory into which the models should be saved.
        times: {'all', 'realtime'}
            Which time indices to train the model on. If 'all', a full run (32 timesteps) is used.
            If 'realtime', range(10, 31, 4) is used.
        pred_years: int or None, default=None
            Which year to build models for. If None, the default values from the paper (range(2009, 2016))
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
            In addition to MSE, L1 loss is also used (sometimes). The default is 0, but a value of 1.5 is used
            when training the model in batch
        patience: int or None, default=10
            The number of epochs to wait without improvement in the validation loss before terminating training.
            Note that the original repository doesn't use early stopping.

        use_gp: boolean, default=True
            Whether to use a Gaussian process in addition to the model

        If use_gp=True, the following parameters are also used:

        sigma: float, default=1
            The kernel variance, or the signal variance
        r_loc: float, default=0.5
            The length scale for the location data (latitudes and longitudes)
        r_year: float, default=1.5
            The length scale for the time data (years)
        sigma_e: float, default=0.32
            Noise variance. 0.32 **2 ~= 0.1
        sigma_b: float, default=0.01
            Parameter variance; the variance on B

        device: torch.device
            Device to run model on. By default, checks for a GPU. If none exists, uses
            the CPU

        """
        wandb.init(project='crop_yield_prediction_4', config={"train_steps": train_steps,
                                                              "batch_size": batch_size,
                                                              "starter_learning_rate": starter_learning_rate,
                                                              "dropout": dropout,
                                                              "use_gp": use_gp,
                                                              "pred_years": pred_years,
                                                              })

        histogram_path = Path(cleaned_data_path) / "histogram_all_full.npz"

        study = optuna.create_study(direction="minimize")

        def objective(trial):
            joblib.dump(study, "data/optuna_study_ave_new.pkl")

            model = ConvModel(
                in_channels=9,
                dropout=trial.suggest_float("dropout", 0, 1),
                dense_features=dense_features,
                savedir=savedir,
                use_gp=None,
                sigma=sigma,
                r_loc=r_loc,
                r_year=r_year,
                sigma_e=sigma_e,
                sigma_b=sigma_b,
                device=device,
            )

            rmse, _rsq = model.run(
                histogram_path,
                times,
                pred_years,
                num_runs=1,
                train_steps=trial.suggest_int("train_steps", 10000, 50000, 1000),
                batch_size=32,
                starter_learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2),
                weight_decay=0,     # trial.suggest_int("weighty_decay", 0, 1),
                l1_weight=0,  # trial.suggest_int("l1_weight", 0, 1),
                patience=trial.suggest_int("patience", 0, 50),
                ret=True
            )

            study.trials_dataframe().to_csv("data/optuna_results_usa_absunclean32.csv")

            return rmse

        study.optimize(objective, n_trials=30)
        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)
        wandb.config = {'Best_Trial': study.best_trial.params}
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
        fig2 = optuna.visualization.plot_parallel_coordinate(study)
        fig2.show()
        fig3 = optuna.visualization.plot_param_importances(study)
        fig3.show()


if __name__ == "__main__":
    fire.Fire(RunTask)
