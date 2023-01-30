# -*- coding: utf-8 -*-
"""BertClassifier

The purpose of the module is to finetune a Bert Classifier with a Public Procurement dataset
Contact: Jan Globisz
jan.globisz@studbocconi.it

"""

import os
from torch.nn import functional as F
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
        notice_text = data_row[self.text_columns].values.tolist()
        if (len(self.categorical_columns) == 0) & (len(self.categorical_columns) == 0):
            categorical_features = torch.empty(1)
            numerical_features = torch.empty(1)
        else:
            categorical_features = torch.tensor(
                data_row[self.categorical_columns].values
            )
            numerical_features = torch.tensor(data_row[self.numerical_columns].values)
        if self.num_cat_to_text:

            # Concatenate text, numerical and categorical data SET TO STR TODO
            non_text_cols = data_row[
                self.categorical_columns + self.numerical_columns
            ].apply(", ".join, axis=1)
            non_text_cols_vals = non_text_cols.values.tolist()
            input_data = ", ".join([notice_text, non_text_cols_vals])
        else:
            input_data = notice_text
        labels = torch.tensor(data_row[self.label_columns].astype(int).values)

        encoding = self.tokenizer.encode_plus(
            input_data[0],
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
        non_text_cols=None,
        combine_last_layer=False,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.bert_architecture = bert_architecture
        self.label_columns = label_columns

        if is_local_files(self.bert_architecture):
            self.tokenizer = AutoTokenizer.from_pretrained(
                bert_architecture, local_files_only=True
            )
            self.bert_classifier_auto = (
                AutoModelForSequenceClassification.from_pretrained(
                    bert_architecture, local_files_only=True
                )
            )

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_architecture)
            self.bert_classifier_auto = (
                AutoModelForSequenceClassification.from_pretrained(bert_architecture)
            )
        self.combine_last_layer = combine_last_layer
        self.model = self.bert_classifier_auto.base_model
        if self.bert_architecture == 'distilbert-base-multilingual-cased':
            self.pre_classifier = self.bert_classifier_auto.pre_classifier
        else:
            self.pre_classifier = self.bert_classifier_auto.bert.pooler.Linear
        if self.combine_last_layer:
            self.bert_classifier_auto.classifier.in_features = (
                self.bert.config.hidden_size + len(non_text_cols)
            )
        self.classifiers = [
            self.bert_classifier_auto.classifier for i in range(len(label_columns))
        ]

        self.dropout = self.bert_classifier_auto.dropout

        self.combine_last_layer = combine_last_layer

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

        hidden_state = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        ).hidden_states[
            0
        ]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        if self.bert_architecture.split("-") == "distillbert":
            pooled_output = torch.nn.ReLU()(pooled_output)  # (bs, dim)
        elif self.bert_architecture.split("-") == "bert":
            pooled_output = torch.nn.Tanh()()(pooled_output)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        if self.combine_last_layer:
            pooled_output = torch.cat(pooled_output, non_text)
        output = []
        for i, label in range(len(self.label_columns)):
            logits = self.classifiers[i](pooled_output)  # (bs, num_labels)
            preds = torch.sigmoid(torch.argmax(logits, 1))
            output.append(preds)

        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        categorical_features = batch["categorical_features"]
        numerical_features = batch["numerical_features"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        hidden_state = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        ).hidden_states[
            0
        ]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        if self.bert_architecture.split("-") == "distillbert":
            pooled_output = torch.nn.ReLU()(pooled_output)  # (bs, dim)
        elif self.bert_architecture.split("-") == "bert":
            pooled_output = torch.nn.Tanh()()(pooled_output)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        if self.combine_last_layer:  # TODO
            pooled_output = torch.cat(
                (pooled_output, numerical_features, categorical_features)
            )
        result_preds = []
        total_loss = 0
        for i in range(len(self.label_columns)):
            logits = self.classifiers[i](pooled_output)  # (bs, num_labels)
            preds = torch.sigmoid(torch.argmax(logits, 1))
            target = labels[:, i].unsqueeze(1)
            total_loss += F.cross_entropy(logits, labels[:, i])
            result_preds.append(preds)

        predictions = torch.cat(result_preds, dim=-1)

        self.log("train_loss", total_loss, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": total_loss, "predictions": predictions, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        categorical_features = batch["categorical_features"]
        numerical_features = batch["numerical_features"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        hidden_state = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        ).hidden_states[
            0
        ]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        if self.bert_architecture.split("-") == "distillbert":
            pooled_output = torch.nn.ReLU()(pooled_output)  # (bs, dim)
        elif self.bert_architecture.split("-") == "bert":
            pooled_output = torch.nn.Tanh()()(pooled_output)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        if self.combine_last_layer:  # TODO
            pooled_output = torch.cat(
                (pooled_output, numerical_features, categorical_features)
            )
        total_loss = 0
        for i in range(len(self.label_columns)):
            logits = self.classifiers[i](pooled_output)  # (bs, num_labels)
            total_loss += F.cross_entropy(logits, labels[:, i])

        self.log("val_loss", total_loss, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        categorical_features = batch["categorical_features"]
        numerical_features = batch["numerical_features"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        hidden_state = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        ).hidden_states[
            0
        ]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = torch.nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        if self.combine_last_layer:
            pooled_output = torch.cat(
                (pooled_output, numerical_features, categorical_features)
            )
        total_loss = 0
        for i in range(len(self.label_columns)):
            logits = self.classifiers[i](pooled_output)  # (bs , num_labels)
            total_loss += F.cross_entropy(logits, labels[:, i])

        self.log("test_loss", total_loss, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        labels = []
        predictions = []

        for output in outputs:
            for out_labels in output["labels"]:
                out_labels = out_labels.detach().cpu()
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                out_predictions = out_predictions.detach().cpu()
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions).reshape(labels.size())

        if self.is_multilabel():
            auroc = AUROC(
                task="multilabel",
                num_labels=len(self.label_columns),
                average="weighted",
            )
            accuracy = Accuracy(
                task="multilabel",
                num_labels=len(self.label_columns),
                average="weighted",
            )
            f1 = F1Score(
                task="multilabel",
                num_labels=len(self.label_columns),
                average="weighted",
            )
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
