import argparse
import logging

from functions import read_and_split, process_data, run_model
from utils import calc_steps
from transformers import (
    AutoTokenizer,
)
RANDOM_SEED = 42

BERT_MODEL_NAME = "distilbert-base-multilingual-cased"
TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)


def main(args):

    logging.info("Reading and processing dataset")
    train_df, test_df = read_and_split(args)
    data_module = process_data(args, train_df, test_df, TOKENIZER)
    total_training_steps, warmup_steps = calc_steps(train_df, args)
    logging.info("Model finetuning start")
    run_model(args, total_training_steps, warmup_steps, data_module)
    logging.info("Finetuning complete!")


if __name__ == "__main__":
    logging.basicConfig(
        filename="bertclassifier.log", level=logging.DEBUG, format='%(asctime)s %(message)s'
    )
    logging.info("Program start")
    logging.info("Collecting arguments")
    parser = argparse.ArgumentParser(
        prog="BertClassifer", description="Finetune using DeepVacuity"
    )
    parser.add_argument("data_path", type=str, help="path to the dataset")

    parser.add_argument(
        "label_column",
        type=str,
        help="Column with the data label (e.g. fraud/no fraud)",
    )
    parser.add_argument(
        "text_column",
        type=str,
        help="Column with the text data (e.g. procurement description, etc.)",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="checkpoints",
        type=str,
        help="path to save model checkpoints",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size - change if needed (reduce for lower memory usage)",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs to run  - change if needed",
    )
    parser.add_argument(
        "--max_token_count",
        type=int,
        default=256,
        help="Max number of tokens to use in Bert model",
    )
    parser.add_argument(
        "--gpu",
        type=bool,
        default=True,
        help="Use gpu for training (strongly recommended)",
    )
    args = parser.parse_args()
    main(args)
