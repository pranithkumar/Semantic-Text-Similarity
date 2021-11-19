import torch.nn as nn
from sklearn import metrics
import argparse

from bertmodel import BERTClassification
from data import *
from test import eval_func

import random

seed_val = 1337

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


if __name__ == '__main__':
    # Setup parser arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--test_data_file_path', type=str, default='data/snli_1.0/snli_1.0_test.jsonl',
                        help='test data file path')
    parser.add_argument('--model_path', type=str, default='models/model.bin', help='model path to load')
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

    # Load pre-trained glove embeddings
    if args.include_glove:
        train_instances = read_instances(args.train_data_file_path, MAX_NUM_TOKENS)
        with open('data/glove_common_words.txt', encoding='utf-8') as file:
            glove_common_words = [line.strip() for line in file.readlines() if line.strip()]
        vocab_token_to_id, vocab_id_to_token = build_vocabulary(train_instances, VOCAB_SIZE, glove_common_words)
        embeddings = load_glove_embeddings('data/glove.6B.' + str(EMBEDDING_DIM) + 'd.txt', EMBEDDING_DIM, vocab_id_to_token)
        embedding_layer = nn.Embedding.from_pretrained(torch.Tensor(embeddings).to(device), freeze=True)
        train_instances = index_instances(train_instances, vocab_token_to_id)
        save_vocabulary(vocab_id_to_token, 'models/vocab.txt')
        df = pd.DataFrame(train_instances)
        df_test = pd.DataFrame({
            'sentence1': df['sentence1'],
            'sentence2': df['sentence2'],
            'gold_label': df['gold_label'],
            'sentence1_token_ids': df['sentence1_token_ids'],
            'sentence2_token_ids': df['sentence2_token_ids']
        })
    else:
        df = pd.read_json(args.train_data_file_path, lines=True)
        df_test = pd.DataFrame({
            'sentence1': df['sentence1'],
            'sentence2': df['sentence2'],
            'gold_label': df['gold_label']
        })

    df_test['contradiction'] = np.where(df_test['gold_label'] == 'contradiction', 1, 0)
    df_test['neutral'] = np.where(df_test['gold_label'] == 'neutral', 1, 0)
    df_test['entailment'] = np.where(df_test['gold_label'] == 'entailment', 1, 0)
    df_test['label'] = df_test[['contradiction', 'neutral', 'entailment']].values.tolist()
    df_test.drop(['contradiction', 'neutral', 'entailment'], axis=1)

    df_test = df_test.reset_index(drop=True)

    test_dataset = DATALoader(data=df_test, max_length=512, glove_model=args.include_glove)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    model = BERTClassification(bert_trainable=args.bert_trainable, glove_model=args.include_glove, embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    outputs, targets = eval_func(data_loader=test_data_loader, model=model, device=device)
    accuracy = metrics.accuracy_score(targets.argmax(1), outputs.argmax(1))
    print(f"Accuracy Score: {accuracy}")