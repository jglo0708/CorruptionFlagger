# -*- coding: utf-8 -*-
"""BertClassifier
The purpose of the module is to finetune a Bert Classifer with a Public Procurement dataset
Contact: Jan Globisz
jan.globisz@studbocconi.it

"""

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray import tune

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
        # num_cat_to_text=args.num_cat_to_text,
    )

    return data_module


def run_model(args, warmup_steps, total_training_steps, data_module, labels):
    """

    :param args: command-line arguments
    :param warmup_steps: number of warmup steps
    :param total_training_steps: calculated total training steps
    :param data_module: Lightning Pytorch data module object
    :return: None
    """
    # make directory to same checkpoints
    if is_local_files(args.bert_architecture):
        dir_path = os.path.join(args.checkpoint_path, "custom_fine_tuned")
    else:
        dir_path = os.path.join(
            args.checkpoint_path, str(args.bert_architecture).split("-")[0]
        )
    os.makedirs(dir_path, exist_ok=True)

    model = ProcurementFlagsTagger(
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
        bert_architecture=args.bert_architecture,
        learning_rate=args.learning_rate,
        label_columns=labels,
        combine_last_layer=args.combine_last_layer,
        non_text_cols=data_module.numerical_columns + data_module.categorical_columns,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_path,
        save_last=True,
        save_top_k=1,
        verbose=True,
        filename="PL--{epoch}-{val_loss:.3f}-{train_loss:.3f}",
        monitor="val_loss",
        mode="min",
    )
    logger = TensorBoardLogger(
        "lightning_logs", name="corruption_indicators", log_graph=True
    )
    logger.log_hyperparams(
        {
            "epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "model": args.bert_architecture,
        }
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)
    if args.resume_from_checkpoint is not None:
        trainer = pl.Trainer(
            logger=logger,
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=args.n_epochs,
            resume_from_checkpoint=args.resume_from_checkpoint,
            accelerator="gpu",
            # strategy='dp',
        )
    else:
        trainer = pl.Trainer(
            logger=logger,
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=args.n_epochs,
            accelerator="gpu",
            # strategy='dp',
        )
    trainer.fit(model, data_module)

    if args.run_test:
        # test on the dataset in-distribution
        trainer.test(datamodule=data_module, ckpt_path="best")

    if args.save_transformers_model:
        transformers_path = os.path.join(dir_path, "HF_saved")
        os.makedirs(transformers_path, exist_ok=True)
        #  Save the tokenizer and the backbone LM with HuggingFace's serialization.
        #  To avoid mixing PL's and HuggingFace's serialization:
        #  https://github.com/PyTorchLightning/pytorch-lightning/issues/3096#issuecomment-686877242
        best_model = ProcurementFlagsTagger.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        best_model.get_backbone().save_pretrained(transformers_path)
        best_model.tokenizer.save_pretrained(transformers_path)


def train_flagger_tune(args, config, warmup_steps,total_training_steps, dir_path, data_module, num_epochs=10, num_gpus=2,):

    model = ProcurementFlagsTagger(
        config = config,
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
        bert_architecture=args.bert_architecture,
        label_columns=data_module.label_columns,
        non_text_cols=data_module.numerical_columns + data_module.categorical_columns,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_path,
        save_last=True,
        save_top_k=1,
        verbose=True,
        filename="PL--{epoch}-{val_loss:.3f}-{train_loss:.3f}",
        monitor="val_loss",
        mode="min",
    )
    logger = TensorBoardLogger(
        "lightning_logs", name="corruption_indicators", log_graph=True
    )
    logger.log_hyperparams(
        {
            "epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "model": args.bert_architecture,
        }
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)
    if args.resume_from_checkpoint is not None:
        trainer = pl.Trainer(
            logger=logger,
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=args.n_epochs,
            resume_from_checkpoint=args.resume_from_checkpoint,
            accelerator="gpu",
            # strategy='dp',
        )
    else:
        trainer = pl.Trainer(
            logger=logger,
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs=args.n_epochs,
            accelerator="gpu",
            # strategy='dp',
        )
    trainer.fit(model, data_module)


def tune_flagger_asha(args, data_module, warmup_steps,total_training_steps, num_epochs=10, gpus_per_trial=2):
    # make directory to same checkpoints
    if is_local_files(args.bert_architecture):
        dir_path = os.path.join(args.checkpoint_path, "custom_fine_tuned")
    else:
        dir_path = os.path.join(
            args.checkpoint_path, str(args.bert_architecture).split("-")[0]
        )
    os.makedirs(dir_path, exist_ok=True)
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        # "batch_size": tune.choice([8, 16, 32]),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "combine_features": tune.choice([True, False]),
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=[ "lr",'weight_decay','combine_features'],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train_flagger_tune,
                                                    num_epochs=num_epochs,
                                                    warmup_steps = warmup_steps,
                                                    total_training_steps = total_training_steps,
                                                    dir_path = dir_path,
                                                    data_module = data_module,
                                                    num_gpus=2,)
                                                    # num_gpus=gpus_per_trial,

    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}


    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=config,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_corruption_flagger_asha")

    print("Best hyperparameters found were: ", analysis.best_config)

    # if args.run_test:
    #     # test on the dataset in-distribution
    #     trainer.test(datamodule=data_module, ckpt_path="best")
    #
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






