# -*- coding: utf-8 -*-
"""BertClassifier

The purpose of the module is to finetune a Bert Classifier with a Public Procurement dataset
Contact: Jan Globisz
jan.globisz@studbocconi.it

"""

import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC, Accuracy, F1Score
from transformers import (
    AdamW,
    # AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pytorch_lightning as pl

MAX_TOKEN_COUNT = 256
RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED)

BERT_MODEL_NAME = "distilbert-base-multilingual-cased"
TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)


class ProcurementNoticeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_token_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.df = df
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        data_row = self.df.df.iloc[index]
        notice_text = data_row.text
        labels = torch.tensor(int(data_row["label_encoded"])).long()

        encoding = self.tokenizer.encode_plus(
            notice_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return dict(
            notice_text=notice_text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=labels
        )


class ProcurementNoticeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        test_df,
        tokenizer,
        batch_size=16,
        max_token_len=256,
    ):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage = None):
        self.train_dataset = ProcurementNoticeDataset(
            self.train_df, self.tokenizer, self.max_token_len
        )

        self.val_dataset = ProcurementNoticeDataset(
            self.val_df, self.tokenizer, self.max_token_len
        )

        self.test_dataset = ProcurementNoticeDataset(
            self.test_df, self.tokenizer, self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)


class ProcurementFlagsTagger(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        label_column: str,
        n_training_steps=None,
        n_warmup_steps=None,
    ):

        super().__init__()
        self.n_classes = n_classes
        self.bert_classifier = AutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL_NAME
        )
        self.label_column = label_column
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert_classifier(input_ids = input_ids, attention_mask=attention_mask, labels = labels)
        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output = self(input_ids, attention_mask, labels)
        loss = output.loss
        preds = torch.sigmoid(output.logits)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": preds, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output = self(input_ids, attention_mask, labels)
        loss = output.loss
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output = self(input_ids, attention_mask, labels)
        loss = output.loss
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(self.label_column):
            auroc = AUROC(num_classes=self.n_classes)
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(
                f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch
            )

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )
