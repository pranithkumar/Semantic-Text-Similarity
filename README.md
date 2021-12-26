# Fall 2021 CSE 538 - Project
#### Krishna Teja Reddy Chinnakotla - 114507668
#### Pranith Kumar Raparthi - 113217634
#### Varun Doniparti - 113270819

The google drive link in the report contains the below folder structure where the models are stored in individual folders for each scenario in models folder and the datasets are stored in the data folder.
```
semanticTextSimilarity
├── checkpoints/
├── data/
├── models/
├── bertmodel.py
├── data.py
├── main.py
├── predict.py
├── test.py
├── train.py
├── README.md
```

`data.py` - The functions to load glove embeddings, load and save vocabulary were taken from the homework assignments. We have updated the generate_batches (which calls the get_token) function as per our requirement and included the save_ckp and load_ckp functions. Also, the read instance function was modified as per our requirement. 

`bertmodel.py` - This file consists of model classes for bert and sbert which are tweaked as needed for the project.

## Example Commands
There are two scripts to run in the project `train.py` and `predict.py` for training and predicting the bert/sbert Models. You can supply `-h` flag to know about the command line arguments for each script.

Below are the sample commands.

#### Train a model
```
python main.py --model sbert --bert_trainable --suffix_name train_sbert_witout_glove_mednli
python main.py --model sbert --bert_trainable --include_glove --suffix_name train_sbert_with_glove_mednli

# stores the model by default at : models/default
```

#### Predict with model
```
python predict.py --model_folder models/train_sbert_witout_glove_mednli --dataset_name mednli
python predict.py --model_folder models/train_sbert_wit_glove_mednli --dataset_name mednli
```

#### Required packages
Download Spacy package
```
python3 -m spacy download en_core_web_sm
```
Below are some of the required packages to be installed
```
nltk
numpy
scikit-learn
scipy
sentence-transformers
sentencepiece
torch
tqdm
transformers
```