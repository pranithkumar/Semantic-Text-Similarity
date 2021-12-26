# Python libraries
import numpy as np
import pandas as pd
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn import metrics
import argparse, os

# Project libraries
from bertmodel import BERTClassification, SBERTClassification
from data import *
from test import eval_func
from train import train_func
from scipy.stats import pearsonr

seed_val = 1337

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

if __name__ == '__main__':
    # Setup parser arguments
    parser = argparse.ArgumentParser(description='Train bert/sbert Model')
    parser.add_argument('--model', type=str, default='sbert', choices=('bert', 'sbert', 'sbert_cls', 'sbert_layer1'), help='There are four setups for the model type which include bert, sbert (sbert with mean pooling), sbert_cls (use the sbert CLS token output), sbert_layer1 (concatenate sbert with mean pooling output with sbert layer1 output before passing to output layer).')
    parser.add_argument('--max_tokens', type=str, default=512, help='Maximum number of tokens')
    parser.add_argument('--bert_trainable', action="store_true", default=False, help='pass this argument to unfreeze bert layers')
    parser.add_argument('--include_glove', action="store_true", default=False, help='pass this for the model to concatenate glove embeddings with the bert/sbert output before passing it to the output layer.')
    parser.add_argument('--vocab_size', type=int, default=10000, help='vocabulary size')
    parser.add_argument('--emb_size', type=int, default=50, help='embedding size')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='max num epochs to train for')
    parser.add_argument('--suffix_name', type=str, default="default",
                        help='optional model name suffix. can be used to prevent name conflict '
                             'in experiment output models directory')
    parser.add_argument('--dataset_name', type=str,default="mednli",choices=('snli', 'mednli', 'medSTS'),  help='dataset name to be trained on')
    parser.add_argument('--retrain_model_name', type=str, default=None,
                        help='model name to be loaded for retraining on new dataset')
    args = parser.parse_args()
    if args.model.startswith('bert'):
      args.pretrained_model_name = 'albert-base-v2'
    elif args.model.startswith('sbert'):
      args.pretrained_model_name = 'sentence-transformers/all-distilroberta-v1'


    save_model_dir = os.path.join("models", args.suffix_name)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Set max number of tokens and vocab size
    MAX_NUM_TOKENS = args.max_tokens
    VOCAB_SIZE = args.vocab_size
    EMBEDDING_DIM = args.emb_size
    embedding_layer = None

    # Load dataset
    if args.dataset_name == 'snli':
        data = read_instances('data/snli_1.0/snli_1.0_train.jsonl', MAX_NUM_TOKENS, glove_model=args.include_glove)
        data = data[:int(len(data) * (25 / 100))]
    elif args.dataset_name == 'medSTS':
        data = read_instances_medSTS('data/medSTS/ClinicalSTS/clinicalSTS.train.txt', MAX_NUM_TOKENS, glove_model=args.include_glove)
    elif args.dataset_name == 'mednli':
        data = read_instances_mednil('data/mednli/train.tsv', MAX_NUM_TOKENS, glove_model=args.include_glove)

    # Load pre-trained glove embeddings
    if args.include_glove:
        with open('data/glove_common_words.txt', encoding='utf-8') as file:
            glove_common_words = [line.strip() for line in file.readlines() if line.strip()]
        vocab_token_to_id, vocab_id_to_token = build_vocabulary(data, VOCAB_SIZE, glove_common_words)
        embeddings = load_glove_embeddings('data/glove.6B.' + str(EMBEDDING_DIM) + 'd.txt', EMBEDDING_DIM,
                                           vocab_id_to_token)
        embedding_layer = nn.Embedding.from_pretrained(torch.Tensor(embeddings).to(device), freeze=True)
        train_instances = index_instances(data, vocab_token_to_id)
        save_vocabulary(vocab_id_to_token, os.path.join(save_model_dir, 'vocab.txt'))

    data.sort(key=lambda x: len(x['sentence1']) + len(x['sentence2']), reverse=True)
    df_train, df_valid = train_test_split(data, test_size=0.1, random_state=seed_val)

    print("Reading training data")
    train_data_loader = generate_batches(df_train, batch_size=args.batch_size, device=device,
                                         glove_model=args.include_glove, model_name=args.model, pretrained_model_name = args.pretrained_model_name)
    print("Reading validation data")
    val_data_loader = generate_batches(df_valid, batch_size=args.batch_size, device=device,
                                       glove_model=args.include_glove, model_name=args.model, pretrained_model_name = args.pretrained_model_name)
    if args.model.startswith('bert'):
        model = BERTClassification(bert_trainable=args.bert_trainable, glove_model=args.include_glove, embedding_dim=EMBEDDING_DIM, pretrained_model_name =args.pretrained_model_name, model_name=args.model)
    elif args.model.startswith('sbert'):
        model = SBERTClassification(bert_trainable=args.bert_trainable, glove_model=args.include_glove,
                                   embedding_dim=EMBEDDING_DIM, pretrained_model_name =args.pretrained_model_name, model_name=args.model)
    
    if args.retrain_model_name is not None:
      model_data = torch.load(os.path.join(args.retrain_model_name, "model.bin"))
      model.load_state_dict(model_data['model'])
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm,bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    num_train_steps = len(train_data_loader) * args.num_epochs

    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    best_accuracy = 0
    for epoch in range(args.num_epochs):
        train_func(data_loader=train_data_loader, model=model, optimizer=optimizer, device=device,
                   scheduler=scheduler, embedding_layer=embedding_layer, glove_model=args.include_glove)
        checkpoint = {
            'epoch': epoch + 1,
            'args': args,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        save_ckp(state=checkpoint, checkpoint_dir='checkpoints')
        outputs, targets = eval_func(data_loader=val_data_loader, model=model, device=device,
                                    embedding_layer=embedding_layer, glove_model=args.include_glove)

        accuracy = metrics.accuracy_score(targets.argmax(1), outputs.argmax(1))
        print(f"Accuracy Score: {accuracy}")
        if accuracy > best_accuracy:
            model_parameters = {
              'args': args,
              'model': model.state_dict()
            }
            torch.save(model_parameters, os.path.join(save_model_dir ,"model.bin"))
            best_accuracy = accuracy
