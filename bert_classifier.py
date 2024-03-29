# -*- coding: utf-8 -*-
"""BertClassifier

The purpose of the module is to finetune a Bert Classifier with a Public Procurement dataset
Contact: Jan Globisz
jan.globisz@studbocconi.it

"""

import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC, Accuracy, F1Score
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from utils import is_local_files

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pytorch_lightning as pl


class ProcurementNoticeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        bert_architecture: str = "distilbert-base-multilingual-cased",

        max_sequence_len: int = 256,
    ):
        if is_local_files(bert_architecture):
            self.tokenizer = AutoTokenizer.from_pretrained(
                bert_architecture, local_files_only=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_architecture)

        self.df = df
        self.max_sequence_len = max_sequence_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        data_row = self.df.iloc[index]
        notice_text = data_row.text
        labels = torch.tensor(int(data_row["label_encoded"])).long()

        encoding = self.tokenizer.encode_plus(
            notice_text,
            add_special_tokens=True,
            max_length=self.max_sequence_len,
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
            labels=labels,
        )


class ProcurementNoticeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        test_df,
        batch_size: int = 16,

        max_sequence_len: int = 256,

        bert_architecture: str = "distilbert-base-multilingual-cased",
    ):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.bert_architecture = bert_architecture

        self.max_sequence_len = max_sequence_len

    def setup(self, stage=None):
        self.train_dataset = ProcurementNoticeDataset(
            self.train_df, self.bert_architecture, self.max_sequence_len
        )

        self.val_dataset = ProcurementNoticeDataset(
            self.val_df, self.bert_architecture, self.max_sequence_len
        )

        self.test_dataset = ProcurementNoticeDataset(
            self.test_df, self.bert_architecture, self.max_sequence_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)


class ProcurementFlagsTagger(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        label_column: str,
        bert_architecture: str,
        learning_rate: float,
        n_training_steps=None,
        n_warmup_steps=None,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.bert_architecture = bert_architecture

        if is_local_files(self.bert_architecture):
            self.tokenizer = AutoTokenizer.from_pretrained(
                bert_architecture, local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                bert_architecture, local_files_only=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_architecture)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                bert_architecture
            )


        self.label_column = label_column
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.learning_rate = learning_rate


    def get_backbone(self):
        return self.model

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(

            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output = self(input_ids, attention_mask, labels)
        loss = output.loss
        preds = torch.sigmoid(torch.argmax(output.logits, 1))

        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "predictions": preds, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output = self(input_ids, attention_mask, labels)
        loss = output.loss
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output = self(input_ids, attention_mask, labels)
        loss = output.loss
        self.log("test_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        auroc = AUROC(task="binary")
        accuracy = Accuracy(task="binary")
        f1 = F1Score(task="binary")

        class_roc_auc = auroc(predictions, labels)
        accuracy_score = accuracy(predictions, labels)
        f1_score = f1(predictions, labels)

        self.logger.experiment.add_scalar(
            f"ROC/Train", class_roc_auc, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            f"Accuracy/Train", accuracy_score, self.current_epoch
        )
        self.logger.experiment.add_scalar(f"Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(f"F1/Train", f1_score, self.current_epoch)

    def validation_epoch_end(self, val_outputs):

        avg_loss = torch.stack([x for x in val_outputs]).mean()
        self.logger.experiment.add_scalar(f"Loss/Val", avg_loss, self.current_epoch)

    def test_epoch_end(self, test_outputs):

        avg_loss = torch.stack([x for x in test_outputs]).mean()

        self.logger.experiment.add_scalar(f"Loss/Test", avg_loss, self.current_epoch)


    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )
