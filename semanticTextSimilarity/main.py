# Python libraries
import numpy as np
import pandas as pd
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn import metrics
import argparse

# Project libraries
from bertmodel import BERTClassification, SBERTClassification
from data import *
from test import eval_func
from train import train_func

seed_val = 1337

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

if __name__ == '__main__':
    # Setup parser arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, default='sbert', help='model to train')
    parser.add_argument('--train_data_file_path', type=str, default='data/snli_1.0/snli_1.0_dev.jsonl',
                        help='training data file path')
    parser.add_argument('--max_tokens', type=str, default=512, help='Maximum number of tokens')
    parser.add_argument('--bert_trainable', type=bool, default=True, help='Train bert model')
    parser.add_argument('--include_glove', type=bool, default=False, help='Train bert model')
    parser.add_argument('--vocab_size', type=int, default=10000, help='vocabulary size')
    parser.add_argument('--emb_size', type=int, default=50, help='embedding size')
    parser.add_argument('--batch_size', type=int, default=15, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='max num epochs to train for')
    parser.add_argument('--suffix_name', type=str, default="",
                        help='optional model name suffix. can be used to prevent name conflict '
                             'in experiment output serialization directory')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Set max number of tokens and vocab size
    MAX_NUM_TOKENS = args.max_tokens
    VOCAB_SIZE = args.vocab_size
    EMBEDDING_DIM = args.emb_size
    embedding_layer = None
    print(args.include_glove)
    # Load pre-trained glove embeddings
    # vocab_id_to_token, data = get_data_frame(args.train_data_file_path, args.include_glove, MAX_NUM_TOKENS, VOCAB_SIZE, EMBEDDING_DIM, device)
    if args.include_glove:
        train_instances = read_instances(args.train_data_file_path, MAX_NUM_TOKENS)
        with open('data/glove_common_words.txt', encoding='utf-8') as file:
            glove_common_words = [line.strip() for line in file.readlines() if line.strip()]
        vocab_token_to_id, vocab_id_to_token = build_vocabulary(train_instances, VOCAB_SIZE, glove_common_words)
        embeddings = load_glove_embeddings('data/glove.6B.' + str(EMBEDDING_DIM) + 'd.txt', EMBEDDING_DIM,
                                           vocab_id_to_token)
        embedding_layer = nn.Embedding.from_pretrained(torch.Tensor(embeddings).to(device), freeze=True)
        train_instances = index_instances(train_instances, vocab_token_to_id)
        save_vocabulary(vocab_id_to_token, 'models/vocab.txt')
        df = pd.DataFrame(train_instances)
        data = pd.DataFrame({
            'sentence1': df['sentence1'],
            'sentence2': df['sentence2'],
            'gold_label': df['gold_label'],
            'sentence1_token_ids': df['sentence1_token_ids'],
            'sentence2_token_ids': df['sentence2_token_ids']
        })
    else:
        df = pd.read_json(args.train_data_file_path, lines=True)
        data = pd.DataFrame({
            'sentence1': df['sentence1'],
            'sentence2': df['sentence2'],
            'gold_label': df['gold_label']
        })
    data = data.head(int(len(data) * (1 / 100)))
    # data = data.head(20)
    data['contradiction'] = np.where(data['gold_label'] == 'contradiction', 1, 0)
    data['neutral'] = np.where(data['gold_label'] == 'neutral', 1, 0)
    data['entailment'] = np.where(data['gold_label'] == 'entailment', 1, 0)
    data['label'] = data[['contradiction', 'neutral', 'entailment']].values.tolist()
    data.drop(['contradiction', 'neutral', 'entailment'], axis=1)

    df_train, df_valid = train_test_split(data, test_size=0.1, random_state=seed_val)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = DATALoader(data=df_train, max_length=512, glove_model=args.include_glove, model_name=args.model)
    val_dataset = DATALoader(data=df_valid, max_length=512, glove_model=args.include_glove, model_name=args.model)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1)
    if args.model == 'albert':
        model = BERTClassification(bert_trainable=args.bert_trainable, glove_model=args.include_glove, embedding_dim=EMBEDDING_DIM)
    elif args.model == 'sbert':
        model = SBERTClassification(bert_trainable=args.bert_trainable, glove_model=args.include_glove,
                                   embedding_dim=EMBEDDING_DIM)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm,bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    num_train_steps = len(train_data_loader) * 5

    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    # model, optimizers, scheduler, args.num_epochs = load_ckp(model=model, optimizer=optimizer, scheduler=scheduler,
    #                                                          device=device, checkpoint_path='checkpoints/checkpoint.pt')

    best_accuracy = 0
    for epoch in range(args.num_epochs):
        train_func(data_loader=train_data_loader, model=model, optimizer=optimizer, device=device,
                   scheduler=scheduler, embedding_layer=embedding_layer, glove_model=args.include_glove)
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        save_ckp(state=checkpoint, is_best=True, checkpoint_dir='checkpoints', best_model_dir='models')
        outputs, targets = eval_func(data_loader=val_data_loader, model=model, device=device,
                                    embedding_layer=embedding_layer, glove_model=args.include_glove)
        accuracy = metrics.accuracy_score(targets.argmax(1), outputs.argmax(1))
        print(f"Accuracy Score: {accuracy}")

        if accuracy > best_accuracy:
            torch.save(model.state_dict(), "models/model.bin")
            best_accuracy = accuracy
