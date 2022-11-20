# -*- coding: utf-8 -*-
"""BertClassifier
The purpose of the module is to finetune a Bert Classifer with a Public Procurement dataset
Contact: Jan Globisz
jan.globisz@studbocconi.it

"""

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import pathlib

from bert_classifier import *
from utils import *

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
    df = pd.read_csv(args.data_path)
    assert set(df).issuperset(
        [args.text_column, args.label_column]
    ), f"{args.text_column, args.label_column} need to be valid columns of the dataframe"
    labels_map = dict(
        zip(args.label_column, list(range(0, len(args.label_column) + 1)))
    )
    df["label"] = df[args.label_column].map(labels_map)

    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, seed=RANDOM_SEED)
    return train_df, test_df


def process_data(args, train_df, test_df, tokenizer):

    train_dataset = ProcurementNoticeDataset(
        train_df,
        tokenizer,
        max_token_len=args.max_token_count,
        label_column=args.label_column,
    )

    test_dataset = ProcurementNoticeDataset(
        test_df,
        tokenizer,
        max_token_len=args.max_token_count,
        label_column=args.label_column,
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
        n_classes=len(args.label_columns),
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
        label_column=args.label_column,
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
