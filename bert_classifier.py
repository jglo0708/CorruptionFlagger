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
        max_token_len: int = 256,
        text_columns: list = ["text"],
        categorical_columns: list = [],
        numerical_columns: list = [],
        label_columns: list = ["label_encoded"],
        num_cat_to_text: bool = False,
    ):
        if is_local_files(bert_architecture):
            self.tokenizer = AutoTokenizer.from_pretrained(
                bert_architecture, local_files_only=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_architecture)

        self.df = df
        self.max_token_len = max_token_len
        self.label_columns = label_columns
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.text_columns = text_columns
        self.num_cat_to_text = num_cat_to_text

    def is_multilabel(self):
        if len(self.label_columns) > 1:
            return True
        else:
            return False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        data_row = self.df.iloc[index]
        notice_text = data_row[self.text_columns]
        if (len(self.categorical_columns)==0) &  (len(self.categorical_columns)==0):
            categorical_features = torch.empty((1,1))
            numerical_features =  torch.empty((1,1))
        else:
            categorical_features = data_row[self.categorical_columns]
            numerical_features = data_row[self.numerical_columns]
        if self.num_cat_to_text:

            # Concatenate text, numerical and categorical data
            input_data = torch.cat(
                (notice_text, numerical_features, categorical_features), 0
            )
        else:
            input_data = notice_text
        labels = torch.tensor(int(data_row[self.label_columns])).long()

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
        if self.num_cat_to_text:

            return dict(
                input_data=input_data,
                input_ids=encoding["input_ids"].flatten(),
                attention_mask=encoding["attention_mask"].flatten(),
                labels=labels,
            )
        else:
            return dict(
                input_data=input_data,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
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
        text_columns: list = ["text"],
        categorical_columns: list = None,
        numerical_columns: list = None,
        label_columns: list = ["label_encoded"],
        num_cat_to_text: bool = False,
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
        self.label_columns = label_columns
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.text_columns = text_columns
        self.num_cat_to_text = num_cat_to_text

    def setup(self, stage=None):
        self.train_dataset = ProcurementNoticeDataset(
            df=self.train_df,
            bert_architecture=self.bert_architecture,
            max_token_len=self.max_token_len,
            text_columns=self.text_columns,
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
            label_columns=self.label_columns,
            num_cat_to_text=self.num_cat_to_text,
        )

        self.val_dataset = ProcurementNoticeDataset(
            df=self.val_df,
            bert_architecture=self.bert_architecture,
            max_token_len=self.max_token_len,
            text_columns=self.text_columns,
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
            label_columns=self.label_columns,
            num_cat_to_text=self.num_cat_to_text,
        )

        self.test_dataset = ProcurementNoticeDataset(
            df=self.test_df,
            bert_architecture=self.bert_architecture,
            max_token_len=self.max_token_len,
            text_columns=self.text_columns,
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
            label_columns=self.label_columns,
            num_cat_to_text=self.num_cat_to_text,
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
        label_columns: list,
        bert_architecture: str,
        learning_rate: float,
        n_training_steps=None,
        n_warmup_steps=None,
        non_text_cols = None,
        combine_last_layer = False,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.bert_architecture = bert_architecture
        self.label_columns = label_columns

        if is_local_files(self.bert_architecture):
            self.tokenizer = AutoTokenizer.from_pretrained(
                bert_architecture, local_files_only=True
            )
            self.bert_classifier_auto = AutoModelForSequenceClassification.from_pretrained(
                bert_architecture, local_files_only=True
            )

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                bert_architecture, local_files_only=True
            )
            self.bert_classifier_auto = AutoModelForSequenceClassification.from_pretrained(
                bert_architecture, local_files_only=True
            )

        self.model = self.bert_classifier_auto.base_model
        self.bert_classifier_auto.classifier.out_features = len(self.label_columns)
        self.pre_classifier = self.bert_classifier_auto.pre_classifier
        self.classifiers = [self.bert_classifier_auto.classifier for i in range(len(label_columns))]
        self.dropout = [self.bert_classifier_auto.dropout for i in range(len(label_columns))]

        self.non_text_cols = non_text_cols
        self.combine_last_layer = combine_last_layer
        if self.combine_last_layer:
            self.pre_classifier.in_features = self.bert.config.hidden_size + len(non_text_cols)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.learning_rate = learning_rate

    def is_multilabel(self):
        if len(self.label_columns) > 1:
            return True
        else:
            return False

    def get_backbone(self):
        return self.model

    def forward(self, i, input_ids, attention_mask, non_text, labels):
        embedding = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if self.combine_last_layer:
            embedding = torch.cat(embedding, non_text)

        pre_classifier_input = self.pre_classifier(embedding)
        output = []
        for i in range(len(self.label_columns)):
            logits = self.classifiers[i](pre_classifier_input)
            preds = torch.sigmoid(torch.argmax(logits,1))
            output.append(preds)

        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        categorical_features = batch['categorical_features']
        numerical_features = batch['numerical_features']
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        embedding = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if self.combine_last_layer:
            embedding = torch.cat(embedding, categorical_features, numerical_features)
        pre_classifier_input = self.pre_classifier(embedding)
        total_loss = 0
        preds_list = []
        for i in range(len(self.label_columns)):
            logits = self.classifiers[i](pre_classifier_input)
            preds = torch.sigmoid(torch.argmax(logits, 1))
            total_loss += self.criterion(preds, labels[i])

            preds_list.append(preds)

        result_preds = torch.cat(preds_list, dim=1)

        self.log("train_loss", total_loss, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": total_loss, "predictions": result_preds, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        categorical_features = batch['categorical_features']
        numerical_features = batch['numerical_features']
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        embedding = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if self.combine_last_layer:
            embedding = torch.cat(embedding, categorical_features, numerical_features).last_hidden_state
        pre_classifier_input = self.pre_classifier(embedding)
        total_loss = 0
        for i in range(len(self.label_columns)):
            logits = self.classifiers[i](pre_classifier_input)
            preds = torch.sigmoid(torch.argmax(logits, 1))
            total_loss += self.criterion(preds, labels[i])
        self.log("val_loss", total_loss, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        categorical_features = batch['categorical_features']
        numerical_features = batch['numerical_features']
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        embedding = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if self.combine_last_layer:
            embedding = torch.cat(embedding, categorical_features, numerical_features)
        pre_classifier_input = self.pre_classifier(embedding)
        total_loss = 0
        for i in range(len(self.label_columns)):
            logits = self.classifiers[i](pre_classifier_input)
            preds = torch.sigmoid(torch.argmax(logits, 1))
            total_loss += self.criterion(preds, labels[i])

        self.log("test_loss", total_loss, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

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
        if self.is_multilabel():
            auroc = AUROC(task="multilabel")
            accuracy = Accuracy(task="multilabel")
            f1 = F1Score(task="multilabel")
        else:
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
