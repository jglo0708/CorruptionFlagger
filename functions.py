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

from bert_classifier import ProcurementNoticeDataset, ProcurementNoticeDataModule, ProcurementFlagsTagger
from utils import is_csv

TEST_SIZE = 0.05
RANDOM_SEED = 42


def read_and_split(args):
    '''

    :param args: args from the parser
    :return: test and train Pandas dataframe
    '''

    assert (
        pathlib.Path(args.data_path).resolve().is_file()
    ), f"{args.data_path} should be a file (not dir)"
    assert is_csv(args.data_path), f"{args.data_path} should be a CSV file"
    df = pd.read_csv(args.data_path, sep=None)
    assert set(df.columns).issuperset(
        [args.text_column, args.label_column]
    ), f"{args.text_column, args.label_column} need to be valid columns of the dataframe"
    labels_map = dict(
        zip(df[args.label_column], list(range(0, len(df[args.label_column].unique()) + 1)))
    )
    df["label_encoded"] = df[args.label_column].map(labels_map)

    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    return train_df, test_df


def process_data(args, train_df, test_df, tokenizer):
    train_dataset = ProcurementNoticeDataset(
        data=train_df,
        tokenizer=tokenizer,
        max_token_len=args.max_token_count,
        label_column='label_encoded',
    )

    test_dataset = ProcurementNoticeDataset(
        data=test_df,
        tokenizer=tokenizer,
        max_token_len=args.max_token_count,
        label_column='label_encoded',
    )
    data_module = ProcurementNoticeDataModule(
        train_df=train_dataset,
        test_df=test_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_token_len=args.max_token_count,
    )
    return data_module


def run_model(args, warmup_steps, total_training_steps, data_module):
    model = ProcurementFlagsTagger(
        n_classes=len(data_module.train_df.data[data_module.train_df.label_column].unique()),
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
        label_column='label_encoded',
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
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=args.n_epochs,
        gpus=int(args.gpu),
    )
    trainer.fit(model, data_module)
