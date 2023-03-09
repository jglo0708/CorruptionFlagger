# -*- coding: utf-8 -*-
"""BertClassifier
The purpose of the module is to finetune a Bert Classifer with a Public Procurement dataset
Contact: Jan Globisz
jan.globisz@studbocconi.it

"""

from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from ray.tune.search.optuna import OptunaSearch

from sklearn.model_selection import train_test_split
import pathlib
import pandas as pd
import json
import os

from bert_classifier import (
    ProcurementNoticeDataModule,
    ProcurementFlagsTagger,
)
from utils import is_csv, is_local_files

TEST_SIZE = 0.1
RANDOM_SEED = 42


def read_and_split(args):
    """
    Reads and splits the files into train, val, test sets
    :param args: args from the parser
    :return: test and train Pandas dataframe
    """
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for f in os.listdir(args.data_path):
        file = os.path.join(args.data_path, f)
        assert (
            pathlib.Path(file).resolve().is_file()
        ), f"{file} should be a file (not dir)"
        if args.data_is_json:
            with open(file) as json_file:
                json_data = json.load(json_file)

            train_df_tmp = pd.DataFrame.from_dict(json_data["train"], orient="index")
            val_df_tmp = pd.DataFrame.from_dict(json_data["val"], orient="index")
            test_df_tmp = pd.DataFrame.from_dict(json_data["test"], orient="index")

            train_df = train_df.append(train_df_tmp)
            val_df = val_df.append(val_df_tmp)
            test_df = test_df.append(test_df_tmp)

        elif args.data_is_pkl:
            file_path = file.split("/")[-1].split(".")[0]
            if file_path.split("_")[0] == "train":
                train_df = pd.read_pickle(file)
            elif file_path.split("_")[0] == "val":
                val_df = pd.read_pickle(file)
            elif file_path.split("_")[0] == "test":
                test_df = pd.read_pickle(file)

        else:
            assert is_csv(file), f"{file} should be a CSV file"
            df = pd.read_csv(file, sep=None)
            assert set(df.columns).issuperset(
                [args.text_column, args.label_column]
            ), f"{args.text_column, args.label_column} need to be valid columns of the dataframe"
            labels_map = dict(
                zip(
                    df[args.label_column],
                    list(range(0, len(df[args.label_column].unique()) + 1)),
                )
            )
            df["label_encoded"] = df[args.label_column].map(labels_map)

            train_df_tmp, test_df_tmp = train_test_split(
                df, test_size=TEST_SIZE, random_state=RANDOM_SEED
            )
            val_df_tmp, test_df_tmp = train_test_split(
                test_df_tmp, test_size=0.5, random_state=RANDOM_SEED
            )
            train_df_tmp.append(train_df_tmp)
            val_df.append(val_df_tmp)
            test_df.append(test_df_tmp)

    return train_df, val_df, test_df


def process_data(
    args,
    train_df,
    val_df,
    test_df,
    label_columns,
    text_columns,
    numerical_columns,
    categorical_columns,
):
    """
    Creates a DataModule object
    :param args:
    :param train_df:
    :param val_df:
    :param test_df:
    :return:
    """

    data_module = ProcurementNoticeDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        batch_size=args.batch_size,
        bert_architecture=args.bert_architecture,
        max_token_len=args.max_sequence_len,
        text_columns=text_columns,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        label_columns=label_columns,
    )

    return data_module


def make_dir_path(bert_architecture, checkpoint_path):
    if is_local_files(bert_architecture):
        dir_path = os.path.join(checkpoint_path, "_custom_fine_tuned")
        dir_path = os.path.join(dir_path, str(bert_architecture).split("-")[0])
    else:
        dir_path = os.path.join(checkpoint_path, str(bert_architecture).split("-")[0])
    return dir_path


def train_model(config, args, warmup_steps, total_training_steps, data_module, labels):
    """

    :param config:
    :param labels: label columns given as a list
    :param args: command-line arguments
    :param warmup_steps: number of warmup steps
    :param total_training_steps: calculated total training steps
    :param data_module: Lightning Pytorch data module object
    :return: None
    """
    # make directory to same checkpoints
    dir_path = make_dir_path(args.bert_architecture, args.checkpoint_path)
    os.makedirs(dir_path, exist_ok=True)
    model = ProcurementFlagsTagger(
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
        bert_architecture=args.bert_architecture,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        label_columns=labels,
        combine_last_layer=config['combine_last_layer'],
        non_text_cols=data_module.numerical_columns
        + data_module.train_df[data_module.categorical_columns].columns.tolist(),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_path,
        save_last=True,
        save_top_k=1,
        verbose=True,
        filename="PL--{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
    )
    logger = TensorBoardLogger(
        "lightning_logs", name="corruption_indicators", log_graph=True
    )
    logger.log_hyperparams(
        {
            "epochs": args.n_epochs,
            "model": args.bert_architecture,
        }
    )
    tune_report_callback = TuneReportCallback(
        {"loss": "val_loss"}, on="validation_end"
    )

    callbacks = [checkpoint_callback, tune_report_callback]

    if args.resume_from_checkpoint is not None:
        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            max_epochs=args.n_epochs,
            resume_from_checkpoint=args.resume_from_checkpoint,
            accelerator="gpu",
            # strategy='dp',
        )
    else:
        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            max_epochs=args.n_epochs,
            accelerator="gpu",
        )
    trainer.fit(model, data_module)

    # if args.run_test:
    #     # test on the dataset in-distribution
    #     trainer.test(datamodule=data_module, ckpt_path="last")

    # if args.save_transformers_model:
    #     transformers_path = os.path.join(dir_path, "HF_saved")
    #     os.makedirs(transformers_path, exist_ok=True)
    #     #  Save the tokenizer and the backbone LM with HuggingFace's serialization.
    #     #  To avoid mixing PL's and HuggingFace's serialization:
    #     #  https://github.com/PyTorchLightning/pytorch-lightning/issues/3096#issuecomment-686877242
    #     best_model = ProcurementFlagsTagger.load_from_checkpoint(
    #         checkpoint_callback.best_model_path
    #     )
    #     best_model.get_backbone().save_pretrained(transformers_path)
    #     best_model.tokenizer.save_pretrained(transformers_path)


def tune_corrflagger_asha(
    args,
    warmup_steps,
    total_training_steps,
    data_module,
    labels,
    num_epochs=10,
    gpus_per_trial=1,
):
    config = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.uniform(0, 0.1),
        "combine_last_layer": tune.choice([True, False]),
        # "batch_size": tune.choice([8, 16])
    }
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["combine_last_layer", "learning_rate", "weight_decay"],
        metric_columns=["loss", "f1_score", "training_iteration"],
    )

    train_fn_with_parameters = tune.with_parameters(
        train_model,
        args=args,
        warmup_steps=warmup_steps,
        total_training_steps=total_training_steps,
        data_module=data_module,
        labels=labels,
    )
    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

    tuner = tune.Tuner(
        tune.with_resources(train_fn_with_parameters, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=OptunaSearch(),
            # scheduler=scheduler,
            num_samples=100,
        ),
        run_config=air.RunConfig(
            name="tune_corrflagger_asha",
            progress_reporter=reporter,
            local_dir = '/home/student/jglobisz/CorruptionFlagger'
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)

