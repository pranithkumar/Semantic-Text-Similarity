import torch.nn as nn
from sklearn import metrics
import argparse, os

from bertmodel import BERTClassification, SBERTClassification
from data import *
from test import eval_func
from scipy.stats import pearsonr
import random

seed_val = 1337

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


if __name__ == '__main__':
    # Setup parser arguments
    parser = argparse.ArgumentParser(description='predict on bert/sbert Model')
    parser.add_argument('--model_folder', type=str, required=True, help='Folder name which consists of the model.bin file to be loaded for prediction')
    parser.add_argument('--dataset_name', type=str, required=True, choices=('snli', 'mednli', 'medSTS'), help='dataset name to be used for prediction')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    train_data = torch.load(os.path.join(args.model_folder, "model.bin"))
    
    # Set max number of tokens and vocab size
    MAX_NUM_TOKENS = train_data['args'].max_tokens
    VOCAB_SIZE = train_data['args'].vocab_size
    EMBEDDING_DIM = train_data['args'].emb_size
    embedding_layer = None

    if train_data['args'].model.startswith('bert'):
        model = BERTClassification(bert_trainable=train_data['args'].bert_trainable, glove_model=train_data['args'].include_glove,
                                   embedding_dim=EMBEDDING_DIM, pretrained_model_name=train_data['args'].pretrained_model_name, model_name=train_data['args'].model)
    elif train_data['args'].model.startswith('sbert'):
        model = SBERTClassification(bert_trainable=train_data['args'].bert_trainable, glove_model=train_data['args'].include_glove,
                                    embedding_dim=EMBEDDING_DIM, pretrained_model_name=train_data['args'].pretrained_model_name, model_name=train_data['args'].model)
    
    model.load_state_dict(train_data['model'])
    model.to(device)
    # Load pre-trained glove embeddings
    if args.dataset_name == 'snli':
        data = read_instances('data/snli_1.0/snli_1.0_test.jsonl', MAX_NUM_TOKENS, glove_model=train_data['args'].include_glove)
    elif args.dataset_name == 'medSTS':
        data = read_instances_medSTS_test('data/medSTS/ClinicalSTS/clinicalSTS.test.txt', 'data/medSTS/ClinicalSTS/clinicalSTS.test.gs.sim.txt', MAX_NUM_TOKENS, glove_model=train_data['args'].include_glove)
    elif args.dataset_name == 'mednli':
        data = read_instances_mednil('data/mednli/test.tsv', MAX_NUM_TOKENS, glove_model=train_data['args'].include_glove)

    if train_data['args'].include_glove:
        vocab_token_to_id, vocab_id_to_token = load_vocabulary(os.path.join(args.model_folder,'vocab.txt'))
        embeddings = load_glove_embeddings('data/glove.6B.' + str(EMBEDDING_DIM) + 'd.txt', EMBEDDING_DIM,
                                           vocab_id_to_token)
        embedding_layer = nn.Embedding.from_pretrained(torch.Tensor(embeddings).to(device), freeze=True)
        train_instances = index_instances(data, vocab_token_to_id)

    data.sort(key=lambda x: len(x['sentence1']) + len(x['sentence2']), reverse=True)

    print("Reading test data")
    test_data_loader = generate_batches(data, batch_size=train_data['args'].batch_size, device=device,
                                         glove_model=train_data['args'].include_glove, model_name=train_data['args'].model, pretrained_model_name=train_data['args'].pretrained_model_name)


    outputs, targets = eval_func(data_loader=test_data_loader, model=model, device=device,
                                    embedding_layer=embedding_layer, glove_model=train_data['args'].include_glove)
    accuracy = metrics.accuracy_score(targets.argmax(1), outputs.argmax(1))
    print(f"Accuracy Score: {accuracy}")