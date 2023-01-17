# -*- coding: utf-8 -*-
"""BertClassifier
The purpose of the module is to finetune a Bert Classifer with a Public Procurement dataset
Contact: Jan Globisz
jan.globisz@studbocconi.it

"""

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
import pathlib
import pandas as pd
import json
import os

from bert_classifier import (
    ProcurementNoticeDataset,
    ProcurementNoticeDataModule,
    ProcurementFlagsTagger,
)
from utils import is_csv

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
        print(file)
        assert (
            pathlib.Path(file).resolve().is_file()
        ), f"{file} should be a file (not dir)"
        if not args.data_is_json:
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
        else:
            with open(file) as json_file:
                json_data = json.load(json_file)

            train_df_tmp = pd.DataFrame.from_dict(json_data["train"], orient="index")
            val_df_tmp = pd.DataFrame.from_dict(json_data["val"], orient="index")
            test_df_tmp = pd.DataFrame.from_dict(json_data["test"], orient="index")

            train_df_tmp.rename(
                columns={args.label_column: "label_encoded", args.text_column: "text"},
                inplace=True,
            )
            val_df_tmp.rename(
                columns={args.label_column: "label_encoded", args.text_column: "text"},
                inplace=True,
            )
            test_df_tmp.rename(
                columns={args.label_column: "label_encoded", args.text_column: "text"},
                inplace=True,
            )
            train_df = train_df.append(train_df_tmp)
            val_df = val_df.append(val_df_tmp)
            test_df = test_df.append(test_df_tmp)

    return train_df, val_df, test_df


def process_data(args, train_df, val_df, test_df):
    '''
    Creates a DataModule object
    :param args:
    :param train_df:
    :param val_df:
    :param test_df:
    :return:
    '''
    train_dataset = ProcurementNoticeDataset(
        df=train_df,

        max_token_len=args.max_token_count,
    )
    val_dataset = ProcurementNoticeDataset(
        df=val_df,

        max_token_len=args.max_token_count,
    )

    test_dataset = ProcurementNoticeDataset(
        df=test_df,

        max_token_len=args.max_token_count,
    )
    data_module = ProcurementNoticeDataModule(
        train_df=train_dataset,
        val_df=val_dataset,
        test_df=test_dataset,
        batch_size=args.batch_size,
        max_token_len=args.max_token_count,
    )
    return data_module


def run_model(args, warmup_steps, total_training_steps, data_module):
    """

    :param args: command-line arguments
    :param warmup_steps: number of warmup steps
    :param total_training_steps: calculated total training steps
    :param data_module: Lightning Pytorch data module object
    :return: None
    """
    model = ProcurementFlagsTagger(
        n_classes=2,
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
        label_column="label_encoded",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename="best-checkpoint",
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    logger = TensorBoardLogger("lightning_logs", name="corruption_indicators")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=args.n_epochs,
        accelerator="gpu" if args.gpu else "cpu",
        # strategy='dp',
    )
    trainer.fit(model, data_module)
