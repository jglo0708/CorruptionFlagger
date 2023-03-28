import argparse
import logging

from functions import read_and_split, process_data
from utils import calc_steps, seed_everything, get_cols
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def main(args):
    seed_everything()
    logger.info("Reading and processing dataset")
    train_df, val_df, test_df = read_and_split(args)
    label_columns, text_columns, numerical_columns, categorical_columns = get_cols(args)
    logging.info("Train set size=%s", len(train_df))
    logging.info("Val set size=%s", len(val_df))
    logging.info("Test set size=%s", len(test_df))
    data_module = process_data(
        args,
        train_df,
        val_df,
        test_df,
        label_columns,
        text_columns,
        numerical_columns,
        categorical_columns,
    )
    total_training_steps, warmup_steps = calc_steps(train_df, args)
    logging.info("Model fine-tuning start")

    learning_rates = [1e-4, 1e-5, 1e-6, 1e-7]
    for lr in learning_rates:
        train_model(
            args, lr, total_training_steps, warmup_steps, data_module, label_columns
        )
    logger.info("Fine-tuning complete!")



if __name__ == "__main__":
    logger = logging
    logger.basicConfig(
        filename="bertclassifier.log",
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
    )
    logger.info("Program start")
    logger.info("Collecting arguments")
    parser = argparse.ArgumentParser(
        prog="BertClassifer", description="Finetune using a dataset"
    )
    parser.add_argument("data_path", type=str, help="path to the dataset dir")
    parser.add_argument(
        "--data_is_json", default=False, type=bool, help="is dataset in json format"
    )
    parser.add_argument(
        "--data_is_pkl", default=True, type=bool, help="is dataset in pkl format"
    )

    parser.add_argument(
        "--text_columns_dir",
        default="data_utils/text_cols.txt",
        help="Dir path to categorical columns file",
    )
    parser.add_argument(
        "--label_columns_dir",
        default="data_utils/label_cols.txt",
        help="Dir path to categorical columns file",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="checkpoints",
        type=str,
        help="path to save model checkpoints. in case you are using test mode, please specify directly the checkpoint to use",
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
        default=5,
        help="Number of epochs to run  - change if needed",
    )
    parser.add_argument(
        "--bert_architecture",
        type=str,
        default="distilbert-base-multilingual-cased",
        help="Underlying Bert model",
    )
    parser.add_argument(
        "--max_sequence_len",
        type=int,
        default=256,
        help="Maximum sequence length of tokens to use in Bert model",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for the optimiser",
    )
    parser.add_argument(
        "--categorical_columns_dir",
        default="data_utils/categoricals.txt",
        help="Dir path to categorical columns file",
    )
    parser.add_argument(
        "--numerical_columns_dir",
        type=str,
        default="data_utils/numericals.txt",
        help="Dir path to numerical columns file",
    )

    parser.add_argument(
        "--combine_last_layer",
        type=bool,
        default=False,
        help="Whether we combine the numerical and categorical&numerical features with BERT embedding ",
    )
    parser.add_argument(
        "--save_transformers_model",
        type=bool,
        default=True,
        help="save Transformers model",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="In case you wish to continue training, input the relative path of the .cpt file",
    )
    parser.add_argument(
        "--run_test",
        type=bool,
        default=None,
        help="In case you wish to run the model in test mode",
    )
    args = parser.parse_args()
    main(args)