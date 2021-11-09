# Python libraries
import numpy as np
import pandas as pd
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
from sklearn.model_selection import train_test_split
from sklearn import metrics
import argparse

# Project libraries
from bertmodel import BERTClassification
from data import *
from test import eval_func
from train import train_func
import torch
import torch.nn as nn

if __name__ == '__main__':
    # Setup parser arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--train_data_file_path', type=str, default='data/snli_1.0/snli_1.0_train.jsonl',
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
    with open('data/glove_common_words.txt', encoding='utf-8') as file:
        glove_common_words = [line.strip() for line in file.readlines() if line.strip()]
    vocab_token_to_id, vocab_id_to_token = build_vocabulary(train_instances, VOCAB_SIZE, glove_common_words)
    embeddings = load_glove_embeddings('data/glove.6B.50d.txt', EMBEDDING_DIM, vocab_id_to_token)
    train_instances = index_instances(train_instances, vocab_token_to_id)
    save_vocabulary(vocab_id_to_token, 'models/vocab.txt')

    # Load train dataset
    # df = pd.read_json(args.train_data_file_path, lines=True)
    df = pd.DataFrame(train_instances)
    df = df.head(int(len(df)*(25/100)))
    # df = df.head(2000)
    df['contradiction'] = np.where(df['gold_label'] == 'contradiction', 1, 0)
    df['neutral'] = np.where(df['gold_label'] == 'neutral', 1, 0)
    df['entailment'] = np.where(df['gold_label'] == 'entailment', 1, 0)
    df['merged_labels'] = df[['contradiction', 'neutral', 'entailment']].values.tolist()
    df.drop(['contradiction', 'neutral', 'entailment'], axis=1)

    data = pd.DataFrame({
        'sentence1': df['sentence1'],
        'sentence2': df['sentence2'],
        'label': df['merged_labels'],
        'sentence1_token_ids': df['sentence1_token_ids'],
        'sentence2_token_ids': df['sentence2_token_ids']
    })

    # Split data into training and validation sets
    df_train, df_valid = train_test_split(data, test_size=0.1, random_state=23)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = DATALoader(data1=df_train.sentence1.values, data2=df_train.sentence2.values,
                               target=df_train.label.values, max_length=512,
                               data1_tokens=df_train.sentence1_token_ids.values,
                               data2_tokens=df_train.sentence2_token_ids.values)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=15, num_workers=4)

    val_dataset = DATALoader(data1=df_valid.sentence1.values, data2=df_valid.sentence2.values,
                             target=df_valid.label.values, max_length=512,
                             data1_tokens=df_valid.sentence1_token_ids.values,
                             data2_tokens=df_valid.sentence2_token_ids.values)

    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, num_workers=1)
    # print(len(vocab_token_to_id))
    model = BERTClassification(
        device=device,
        vocab_size=min(VOCAB_SIZE, len(vocab_token_to_id)),
        embedding_dim=EMBEDDING_DIM)

    embedding_layer = nn.Embedding.from_pretrained(torch.Tensor(embeddings).to(device), freeze=True)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm,bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    num_train_steps = int(len(df_train) / 8 * 10)
    optimizers = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(optimizers, num_warmup_steps=0, num_training_steps=num_train_steps)

    best_accuracy = 0
    for epoch in range(5):
        train_func(data_loader=train_data_loader, model=model, optimizer=optimizers, device=device, scheduler=scheduler, embedding_layer=embedding_layer)
        outputs, targets = eval_func(data_loader=val_data_loader, model=model, device=device, embedding_layer=embedding_layer)
        accuracy = metrics.accuracy_score(targets.argmax(1), outputs.argmax(1))
        print(f"Accuracy Score: {accuracy}")

        if accuracy > best_accuracy:
            torch.save(model.state_dict(), "models/model.bin")
            best_accuracy = accuracy
