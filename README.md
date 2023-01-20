# Corruption Flagger
This is a project for fine-tuning  a multilingual BERT based classification model for detecing corruption risk flags.

Set up:
1.  Run  `python3 -m venv venv_cf`
2.  Run `source venv_cf/bin/activate`
3.  Run `pip install -r requirements.txt`

It is required to have installed CUDA on your device!

To run the script, use
`python3 run.py PATH_TO_DATA IS_DATA_JSON LABEL_COLUMN TEXT_COLUMN`
e.g. `python3 run.py inputs/json_dir/ True risco txt`


Finetune using an external dataset

positional arguments:
  * **data_path**:             path to the dataset dir
  * **data_is_json**:           is dataset in json format
  * **label_column**:           Column with the data label (e.g. fraud/no fraud)
  * **text_column**:            Column with the text data (e.g. procurement description, etc.)

optional arguments:
  * **checkpoint_path**: path to save model checkpoints (default: 'checkpoints')
  * **batch_size**:  Batch size - change if needed (reduce for lower memory usage) (default: 32)
  * **n_epochs**:  Number of epochs to run - change if needed (default: 10)
  * **max_sequence_len**: Max number of tokens to use in BERT model (default: 256)
  * **bert_architecture**: Underlying Bert model (default: "distilbert-base-multilingual-cased")
  * **learning_rate**: Learning rate for the optimiser (default: 2e-5)
  * **save_transformers_model**: Save Transformers model (default: True)
  * **resume_from_checkpoint**: In case you wish to continue training, input the relative path of the .cpt file (default: None)
  * **run_test**: In case you wish to run the model in test mode (default: False)