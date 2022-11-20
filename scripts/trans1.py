from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('../analysis/test_desc.csv')

texts = data['text'].values.tolist()
labels =  data['tender_indicator_INTEGRITY_SINGLE_BID'].values.tolist()

rest_texts, test_texts, rest_labels, test_labels = train_test_split(texts, labels, test_size=0.1, random_state=1)
train_texts, dev_texts, train_labels, dev_labels = train_test_split(rest_texts, rest_labels, test_size=0.1,
                                                                    random_state=1)

print("Train size:", len(train_texts))
print("Dev size:", len(dev_texts))
print("Test size:", len(test_texts))

target_names = list(set(labels))
label2idx = {label: idx for idx, label in enumerate(target_names)}
print(label2idx)
#
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
#
# pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('lr', LogisticRegression())
# ])
#
# parameters = {'lr__C': [0.1, 0.5, 1, 2, 5, 10, 100, 1000]}
#
# best_classifier = GridSearchCV(pipeline, parameters, cv=5, verbose=1)
# best_classifier.fit(train_texts, train_labels)
# best_predictions = best_classifier.predict(test_texts)
#
# baseline_accuracy = np.mean(best_predictions == test_labels)
baseline_accuracy = 0.8652482269503546
# print("Baseline accuracy:", baseline_accuracy)
#
# from sklearn.dummy import DummyClassifier
# dummy_clf = DummyClassifier(strategy="most_frequent")
# dummy_clf.fit(train_texts, train_labels)
# dummy_predictions = dummy_clf.predict(test_texts)
# dummy_accuracy = np.mean(dummy_predictions == test_labels)
# print("DummyClassifier accuracy:", dummy_accuracy)

import torch
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

BERT_MODEL = "distilbert-base-multilingual-cased"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)


from transformers import AutoModelForSequenceClassification, AutoModel

model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = len(label2idx))
model.to(device)
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 10


class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_inputs(example_texts, example_labels, label2idx, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""

    input_items = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):

        # Create a list of token ids
        input_ids = tokenizer.encode(text, truncation=True, max_length=10)
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label2idx[label]

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    return input_items


train_features = convert_examples_to_inputs(train_texts, train_labels, label2idx, MAX_SEQ_LENGTH, tokenizer, verbose=0)
dev_features = convert_examples_to_inputs(dev_texts, dev_labels, label2idx, MAX_SEQ_LENGTH, tokenizer)
test_features = convert_examples_to_inputs(test_texts, test_labels, label2idx, MAX_SEQ_LENGTH, tokenizer)
clean_model = AutoModel.from_pretrained(BERT_MODEL).to(device)

from torch.utils.data import TensorDataset, DataLoader

def get_data_loader(features, batch_size, shuffle=True):

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader
#
BATCH_SIZE = 1

train_dataloader = get_data_loader(train_features,  BATCH_SIZE, shuffle=True)
dev_dataloader = get_data_loader(dev_features,  BATCH_SIZE, shuffle=False)
test_dataloader = get_data_loader(test_features,  BATCH_SIZE, shuffle=False)

#
# def evaluate(model, dataloader):
#     model.eval()
#
#     eval_loss = 0
#     nb_eval_steps = 0
#     predicted_labels, correct_labels = [], []
#
#     for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
#         batch = tuple(t.to(device) for t in batch)
#         input_ids, input_mask, segment_ids, label_ids = batch
#         with torch.no_grad():
#             tmp_eval_loss = model(input_ids, attention_mask=input_mask, labels=label_ids).loss
#             logits = model(input_ids, attention_mask=input_mask, labels=label_ids).logits
#             torch.cuda.empty_cache()
#
#         outputs = np.argmax(logits.to('cpu'), axis=1)
#         label_ids = label_ids.to('cpu').numpy()
#
#         predicted_labels += list(outputs)
#         correct_labels += list(label_ids)
#
#         eval_loss += tmp_eval_loss.mean().item()
#         nb_eval_steps += 1
#
#     eval_loss = eval_loss / nb_eval_steps
#
#     correct_labels = np.array(correct_labels)
#     predicted_labels = np.array(predicted_labels)
#
#     return eval_loss, correct_labels, predicted_labels
#
#
# from transformers import get_linear_schedule_with_warmup
# from torch.optim import AdamW
#
# GRADIENT_ACCUMULATION_STEPS = 16
# NUM_TRAIN_EPOCHS = 20
# LEARNING_RATE = 0.1
# WARMUP_PROPORTION = 0.1
# MAX_GRAD_NORM = 5
#
# num_train_steps = int(len(train_dataloader.dataset) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)
# num_warmup_steps = int(WARMUP_PROPORTION * num_train_steps)
#
# param_optimizer = list(model.named_parameters())
# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#     ]
#
# optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
#
# import torch
# import os
# from tqdm import trange
# from tqdm.notebook import tqdm as tqdm
# from sklearn.metrics import classification_report, precision_recall_fscore_support
#
# OUTPUT_DIR = "/tmp/"
# MODEL_FILE_NAME = "pytorch_model.bin"
# PATIENCE = 2
#
# loss_history = []
# no_improvement = 0
# for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
#     torch.cuda.memory_summary()
#     model.train()
#     tr_loss = 0
#     nb_tr_examples, nb_tr_steps = 0, 0
#     for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
#         torch.cuda.empty_cache()
#         batch = tuple(t.to(device) for t in batch)
#         input_ids, input_mask, segment_ids, label_ids = batch
#         # print(torch.cuda.memory_summary(device=device, abbreviated=False))
#         outputs = model(input_ids, attention_mask=input_mask, labels=label_ids)
#         torch.cuda.empty_cache()
#         loss = outputs[0]
#
#         if GRADIENT_ACCUMULATION_STEPS > 1:
#             loss = loss / GRADIENT_ACCUMULATION_STEPS
#
#         loss.backward()
#         tr_loss += loss.item()
#
#         if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
#
#             optimizer.step()
#             optimizer.zero_grad()
#             scheduler.step()
#
#
#     dev_loss, _, _ = evaluate(model, dev_dataloader)
#
#     print("Loss history:", loss_history)
#     print("Dev loss:", dev_loss)
#
#     if len(loss_history) == 0 or dev_loss < min(loss_history):
#         no_improvement = 0
#         model_to_save = model.module if hasattr(model, 'module') else model
#         output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
#         torch.save(model_to_save.state_dict(), output_model_file)
#     else:
#         no_improvement += 1
#
#     if no_improvement >= PATIENCE:
#         print("No improvement on development set. Finish training.")
#         break
#
#     loss_history.append(dev_loss)
#
#     model_state_dict = torch.load(os.path.join(OUTPUT_DIR, MODEL_FILE_NAME), map_location=lambda storage, loc: storage)
#     model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, state_dict=model_state_dict,
#                                                           num_labels=len(target_names))
#     model.to(device)
#
#     model.eval()
#
#     _, train_correct, train_predicted = evaluate(model, train_dataloader)
#     _, dev_correct, dev_predicted = evaluate(model, dev_dataloader)
#     _, test_correct, test_predicted = evaluate(model, test_dataloader)
#
#     print("Training performance:", precision_recall_fscore_support(train_correct, train_predicted, average="micro"))
#     print("Development performance:", precision_recall_fscore_support(dev_correct, dev_predicted, average="micro"))
#     print("Test performance:", precision_recall_fscore_support(test_correct, test_predicted, average="micro"))
#
#     bert_accuracy = np.mean(test_predicted == test_correct)
#
#     print(classification_report(test_correct, test_predicted, target_names=target_names))
#
#     import pandas as pd
#     import matplotlib.pyplot as plt
#
#     df = pd.DataFrame({"accuracy": {"baseline": baseline_accuracy, "BERT": bert_accuracy}})
#     plt.rcParams['figure.figsize'] = (7, 4)
#     df.plot(kind="bar")
#     torch.cuda.empty_cache()