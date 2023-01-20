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
    AutoModel,
    get_linear_schedule_with_warmup,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pytorch_lightning as pl


class ProcurementNoticeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        bert_architecture: str = "distilbert-base-multilingual-cased",
        max_token_len: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_architecture)
        self.df = df
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        data_row = self.df.iloc[index]
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
            labels=labels,
        )


class ProcurementNoticeDatasetMulti(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        bert_architecture: str = "distilbert-base-multilingual-cased",
        max_token_len: int = 256,
        categorical_columns:list = None,
        numerical_columns: list = None,
        label_columns: list = None,
        combine_num_cat: bool = False
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_architecture)
        self.df = df
        self.max_token_len = max_token_len
        self.categorical_columns = categorical_columns
        self.numerical_columns=numerical_columns
        self.label_columns = label_columns
        self.combine_num_cat = combine_num_cat

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        data_row = self.df.iloc[index]
        notice_text = data_row.text
        categorical_features = data_row[self.categorical_columns]
        numerical_features = data_row[self.numerical_columns]

        if self.combine_num_cat:
            # Concatenate text, numerical and categorical data
            input_data = torch.cat((notice_text, numerical_features, categorical_features), 0)
        else:
            input_data = notice_text

        labels = torch.tensor(data_row[self.label_column])

        encoding = self.tokenizer.encode_plus(
            input_data,
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
            labels=labels,
        )

class ProcurementNoticeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        test_df,
        batch_size: int = 16,
        max_token_len: int = 256,
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
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = ProcurementNoticeDataset(
            self.train_df, self.bert_architecture, self.max_token_len
        )

        self.val_dataset = ProcurementNoticeDataset(
            self.val_df, self.bert_architecture, self.max_token_len
        )

        self.test_dataset = ProcurementNoticeDataset(
            self.test_df, self.bert_architecture, self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

class ProcurementNoticeDataModuleMulti(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        test_df,
        batch_size: int = 16,
        max_token_len: int = 256,
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
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = ProcurementNoticeDatasetMulti(
            self.train_df, self.bert_architecture, self.max_token_len
        )

        self.val_dataset = ProcurementNoticeDatasetMulti(
            self.val_df, self.bert_architecture, self.max_token_len
        )

        self.test_dataset = ProcurementNoticeDatasetMulti(
            self.test_df, self.bert_architecture, self.max_token_len
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
        self.n_classes = n_classes
        self.bert_architecture = bert_architecture
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
        output = self.bert_classifier(
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
        self.logger.experiment.add_scalar(f"Loss/Val", avg_loss, self.current_epoch)

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


class ProcurementFlagsTaggerMulti(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        label_columns: list,
        bert_architecture: str,
        learning_rate: float,
        n_training_steps=None,
        n_warmup_steps=None,
        combine_last_layer:bool=False,
        non_text_cols:list=None,
    ):

        super().__init__()
        self.n_classes = n_classes
        self.bert_architecture = bert_architecture
        # self.config = AutoConfig.from_pretrained((bert_architecture)
        self.model = AutoModel.from_pretrained(
            bert_architecture
        )
        self.non_text_cols=non_text_cols
        self.num_extra_dims = len(non_text_cols) #number of nontext cols
        if combine_last_layer:
            size = self.bert.config.hidden_size
        else:
            size = self.bert.config.hidden_size + self.num_extra_dims


        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(size, n_classes)
        self.criterion = torch.nn.BCELoss()
        self.label_columns = label_columns
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.learning_rate = learning_rate

    def get_backbone(self):
        return self.model

    def forward(self, input_ids, attention_mask, non_text=None, labels=None):
        output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        if self.combine_last_layer :

            cls_embeds = output.last_hidden_state[:, 0, :] # [batch size, hidden size]

            inputs = torch.cat((cls_embeds, non_text), dim=-1) # [batch size, hidden size+num extra dims]

        else:
            inputs = output.last_hidden_state[:, 0, :]  # [batch size, hidden size]

        dropout_output = self.dropout(inputs)
        output = self.classifier(dropout_output)
        # output = self.dropout(inputs)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        non_text=None
        if self.combine_last_layer:
            non_text = batch[self.non_text_cols]
        output = self(input_ids, attention_mask, non_text= non_text, labels=labels)
        loss = output.loss
        preds = torch.sigmoid(torch.argmax(output.logits, 1))

        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "predictions": preds, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        non_text=None
        if self.combine_last_layer:
            non_text = batch[self.non_text_cols]
        output = self(input_ids, attention_mask, non_text= non_text, labels=labels)
        loss = output.loss

        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        non_text=None
        if self.combine_last_layer:
            non_text = batch[self.non_text_cols]
        output = self(input_ids, attention_mask, non_text= non_text, labels=labels)
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
        self.logger.experiment.add_scalar(f"Loss/Val", avg_loss, self.current_epoch)

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
