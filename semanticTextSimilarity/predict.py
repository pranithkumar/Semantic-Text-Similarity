import numpy as np
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics


import pandas as pd

from bertmodel import BERTClassification
from data import *
from test import eval_func
from train import train_func

def run():
    # Setup parser arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--train_data_file_path', type=str, default='data/snli_1.0/snli_1.0_test.jsonl',
                        help='training data file path')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--num-epochs', type=int, default=20, help='max num epochs to train for')
    parser.add_argument('--suffix-name', type=str, default="",
                        help='optional model name suffix. can be used to prevent name conflict '
                             'in experiment output serialization directory')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Set max number of tokens and vocab size
    MAX_NUM_TOKENS = 250
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 50

    # Load pre-trained glove embeddings
    train_instances = read_instances(args.train_data_file_path, MAX_NUM_TOKENS)
    vocab_token_to_id, vocab_id_to_token = load_vocabulary('models/vocab.txt')
    embeddings = load_glove_embeddings('data/glove.6B.50d.txt', EMBEDDING_DIM, vocab_id_to_token)
    train_instances = index_instances(train_instances, vocab_token_to_id)

    # Load train dataset
    # df = pd.read_json(args.train_data_file_path, lines=True)
    df = pd.DataFrame(train_instances)
    # df = df.head(150)
    df['contradiction'] = np.where(df['gold_label'] == 'contradiction', 1, 0)
    df['neutral'] = np.where(df['gold_label'] == 'neutral', 1, 0)
    df['entailment'] = np.where(df['gold_label'] == 'entailment', 1, 0)
    df['merged_labels'] = df[['contradiction', 'neutral', 'entailment']].values.tolist()
    df.drop(['contradiction', 'neutral', 'entailment'], axis=1)

    df_test = pd.DataFrame({
        'sentence1': df['sentence1'],
        'sentence2': df['sentence2'],
        'label': df['merged_labels'],
        'sentence1_token_ids': df['sentence1_token_ids'],
        'sentence2_token_ids': df['sentence2_token_ids']
    })

    df_test = df_test.reset_index(drop=True)

    test_dataset = DATALoader(
        data1=df_test.sentence1.values,
        data2=df_test.sentence2.values,
        target=df_test.label.values,
        max_length=512,
        data1_tokens=df_test.sentence1_token_ids.values,
        data2_tokens=df_test.sentence2_token_ids.values
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        num_workers=4,
    )
    embedding_layer = nn.Embedding.from_pretrained(torch.Tensor(embeddings).to(device), freeze=True)
    # print(len(vocab_token_to_id))
    # print(a[0], a[len(vocab_token_to_id) - 1])
    model = BERTClassification(device=device,
        vocab_size=len(vocab_token_to_id),
        embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load("models/model.bin"))
    model.to(device)

    outputs, targets = eval_func(data_loader=test_data_loader, model=model, device=device, embedding_layer=embedding_layer)
    accuracy = metrics.accuracy_score(targets.argmax(1), outputs.argmax(1))
    print(f"Accuracy Score: {accuracy}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/