import csv
import os
import random
import string

import numpy as np
import torch
import os.path
from os import path


def calc_steps(df, args):
    steps_per_epoch = len(df) // args.batch_size
    total_training_steps = steps_per_epoch * args.n_epochs
    warmup_steps = total_training_steps // 5
    return total_training_steps, warmup_steps


def get_label_colums(df, args):
    label_columns = df[args.label_column].unique()
    return label_columns


def is_csv(infile):
    try:
        with open(infile, newline="") as csvfile:
            start = csvfile.read(4096)

            # isprintable does not allow newlines, printable does not allow umlauts...
            if not all([c in string.printable or c.isprintable() for c in start]):
                return False
            dialect = csv.Sniffer().sniff(start)
            return True
    except csv.Error:
        # Could not get a csv dialect -> probably not a csv.
        return False


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_local_files(path):
    if len(path.split("/")) > 1:
        return True
    else:
        return False

def is_local_files(path):
    if len(path.split("/")) > 1:
        return True
    else:
        return False

def get_cols(args):
    if path.exists(args.text_columns_dir):

        with open(args.text_columns_dir) as file:
            text_columns = [line.rstrip() for line in file]
    else:
        text_columns = []
    if path.exists(args.label_columns_dir):
        with open(args.label_columns_dir) as file:
            label_columns = [line.rstrip() for line in file]
    else:
        label_columns = []

    if path.exists(args.numerical_columns_dir):

        with open(args.numerical_columns_dir) as file:
            numerical_columns = [line.rstrip() for line in file]
    else:
        numerical_columns = []
    if path.exists(args.categorical_columns_dir):
        with open(args.categorical_columns_dir) as file:
            categorical_columns = [line.rstrip() for line in file]
    else:
        categorical_columns = []


    return label_columns, text_columns, numerical_columns, categorical_columns