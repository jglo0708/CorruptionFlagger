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
  * **checkpoint_path**:
                        path to save model checkpoints (default: 'checkpoints')
  * **batch_size**:  Batch size - change if needed (reduce for lower memory usage) (default: 8)
  * **n_epochs**:  Number of epochs to run - change if needed (default: 10)
  * **max_token_count**: Max number of tokens to use in BERT model (default: 256)
  * **gpu GPU**: Use gpu for training (strongly recommended) (default: True)


