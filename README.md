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
  data_path             path to the dataset dir
  data_is_json          is dataset in json format
  label_column          Column with the data label (e.g. fraud/no fraud)
  text_column           Column with the text data (e.g. procurement description, etc.)

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
                        path to save model checkpoints
  --batch_size BATCH_SIZE
                        Batch size - change if needed (reduce for lower memory usage)
  --n_epochs N_EPOCHS   Number of epochs to run - change if needed
  --max_token_count MAX_TOKEN_COUNT
                        Max number of tokens to use in Bert model
  --gpu GPU             Use gpu for training (strongly recommended)


