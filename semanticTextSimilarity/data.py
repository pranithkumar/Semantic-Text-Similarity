import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AlbertTokenizer
from collections import Counter
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import numpy as np
import spacy
import json
import shutil
import random
import torch
import torch.nn as nn
seed_val = 1337

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

nlp = spacy.load("en_core_web_sm", disable=['ner', 'tagger', 'parser', 'textcat'])

def read_instances(data_file_path: str,
                   max_allowed_num_tokens: int = 150, glove_model = False) -> List[Dict]:
    """
    Reads raw classification dataset from a file and returns a list
    of dicts where each dict defines an instance.

    Parameters
    ----------
    data_file_path : ``str``
        Path to data to be read.
    max_allowed_num_tokens : ``int``
        Maximum number of tokens allowed in the classification instance.
        :param data_file_path:
        :param max_allowed_num_tokens:
        :param data:
    """
    instances = []
    with open(data_file_path) as file:
        for line in tqdm(file.readlines()):
            instance = json.loads(line.strip())
            instance = {key: instance[key] for key in ["sentence1", "sentence2", "gold_label"]}
            if instance["gold_label"] == "contradiction":
                instance["label"] = [1, 0, 0]
            elif instance["gold_label"] == "neutral":
                instance["label"] = [0, 1, 0]
            elif instance["gold_label"] == "entailment":
                instance["label"] = [0, 0, 1]
            else:
              continue
            if glove_model:
                tokens_sentence1 = [token.text.lower() for token in nlp.tokenizer(instance["sentence1"])][
                                   :max_allowed_num_tokens]
                tokens_sentence2 = [token.text.lower() for token in nlp.tokenizer(instance["sentence2"])][
                                   :max_allowed_num_tokens]
                instance["sentence1_tokens"] = tokens_sentence1
                instance["sentence2_tokens"] = tokens_sentence2
            instances.append(instance)
    return instances
    

def read_instances_STS(data_file_path: str,
                   max_allowed_num_tokens: int = 150, glove_model = False):
    
    instances = []
    with open(data_file_path) as file:
        for line in tqdm(file.readlines()):
            cols = line.split('\t')
            instance = {}
            instance["sentence1"] = cols[5].strip()
            instance["sentence2"] = cols[6].strip()
            # instance["label"] = ((float(cols[4].strip())*2)/5) - 1 
            instance["gold_label"] = float(cols[4])
            if instance["gold_label"] < 2.0:
                instance["label"] = [1, 0, 0]
            elif instance["gold_label"] < 3.0:
                instance["label"] = [0, 1, 0]
            elif instance["gold_label"] <= 5.0 :
                instance["label"] = [0, 0, 1]

            if glove_model:
                tokens_sentence1 = [token.text.lower() for token in nlp.tokenizer(instance["sentence1"])][
                                   :max_allowed_num_tokens]
                tokens_sentence2 = [token.text.lower() for token in nlp.tokenizer(instance["sentence2"])][
                                   :max_allowed_num_tokens]
                instance["sentence1_tokens"] = tokens_sentence1
                instance["sentence2_tokens"] = tokens_sentence2
            instances.append(instance)
    return instances
                

def read_instances_medSTS(data_file_path: str,
                   max_allowed_num_tokens: int = 150, glove_model = False):
    
    instances = []
    with open(data_file_path) as file:
        for line in tqdm(file.readlines()):
            cols = line.split('\t')
            instance = {}
            instance["sentence1"] = cols[0]
            instance["sentence2"] = cols[1]
            instance["gold_label"] = float(cols[2])
            if instance["gold_label"] < 2.0:
                instance["label"] = [1, 0, 0]
            elif instance["gold_label"] < 3.0:
                instance["label"] = [0, 1, 0]
            elif instance["gold_label"] <= 5.0 :
                instance["label"] = [0, 0, 1]
            else:
              continue
            if glove_model:
                tokens_sentence1 = [token.text.lower() for token in nlp.tokenizer(instance["sentence1"])][
                                   :max_allowed_num_tokens]
                tokens_sentence2 = [token.text.lower() for token in nlp.tokenizer(instance["sentence2"])][
                                   :max_allowed_num_tokens]
                instance["sentence1_tokens"] = tokens_sentence1
                instance["sentence2_tokens"] = tokens_sentence2
            instances.append(instance)
    return instances

def read_instances_medSTS_test(data_file_path: str, targets_file_path,
                   max_allowed_num_tokens: int = 150, glove_model = False):
    
    instances = []
    with open(data_file_path) as file:
        for line in tqdm(file.readlines()):
            cols = line.split('\t')
            instance = {}
            instance["sentence1"] = cols[0]
            instance["sentence2"] = cols[1]
            if glove_model:
                tokens_sentence1 = [token.text.lower() for token in nlp.tokenizer(instance["sentence1"])][
                                   :max_allowed_num_tokens]
                tokens_sentence2 = [token.text.lower() for token in nlp.tokenizer(instance["sentence2"])][
                                   :max_allowed_num_tokens]
                instance["sentence1_tokens"] = tokens_sentence1
                instance["sentence2_tokens"] = tokens_sentence2
            instances.append(instance)
    targets = []
    with open(targets_file_path) as file:
        for line in tqdm(file.readlines()):
            cols = line.split('\t')
            targets.append(float(cols[0]))
            
    for i in range(len(instances)):
        instances[i]["gold_label"] = targets[i]
        if instances[i]["gold_label"] < 2.0:
            instances[i]["label"] = [1, 0, 0]
        elif instances[i]["gold_label"] < 3.0:
            instances[i]["label"] = [0, 1, 0]
        elif instances[i]["gold_label"] <= 5.0 :
            instances[i]["label"] = [0, 0, 1]
        else:
            continue
    return instances

def read_instances_mednil(data_file_path: str,
                   max_allowed_num_tokens: int = 150, glove_model = False):
    
    instances = []
    with open(data_file_path) as file:
        for line in tqdm(file.readlines()):
            cols = line.split('\t')
            instance = {}
            sentences = cols[0].split('sentence1: ')[1].split('sentence2: ')
            instance["sentence1"] = sentences[0]
            instance["sentence2"] = sentences[1]
            instance["gold_label"] = cols[1].strip()
            if instance["gold_label"] == "contradiction":
                instance["label"] = [1, 0, 0]
            elif instance["gold_label"] == "neutral":
                instance["label"] = [0, 1, 0]
            elif instance["gold_label"] == "entailment":
                instance["label"] = [0, 0, 1]
            else:
              continue
            if glove_model:
                tokens_sentence1 = [token.text.lower() for token in nlp.tokenizer(instance["sentence1"])][
                                   :max_allowed_num_tokens]
                tokens_sentence2 = [token.text.lower() for token in nlp.tokenizer(instance["sentence2"])][
                                   :max_allowed_num_tokens]
                instance["sentence1_tokens"] = tokens_sentence1
                instance["sentence2_tokens"] = tokens_sentence2
            instances.append(instance)
    return instances

def index_instances(instances: List[Dict], token_to_id: Dict) -> List[Dict]:
    """
    Uses the vocabulary to index the fields of the instances. This function
    prepares the instances to be tensorized.
    """
    for instance in instances:
        token_ids = []
        for token in instance["sentence1_tokens"]:
            if token in token_to_id:
                token_ids.append(token_to_id[token])
            else:
                token_ids.append(1)  # 1 is index for UNK
        instance["sentence1_token_ids"] = token_ids
        instance.pop("sentence1_tokens")
        token_ids = []
        for token in instance["sentence2_tokens"]:
            if token in token_to_id:
                token_ids.append(token_to_id[token])
            else:
                token_ids.append(1)  # 1 is index for UNK
        instance["sentence2_token_ids"] = token_ids
        instance.pop("sentence2_tokens")
    return instances


def build_vocabulary(instances: List[Dict],
                     vocab_size: 10000,
                     add_tokens: List[str] = None) -> Tuple[Dict, Dict]:
    """
    Given the instances and max vocab size, this function builds the
    token to index and index to token vocabularies. If list of add_tokens are
    passed, those words will be added first.

    Parameters
    ----------
    instances : ``List[Dict]``
        List of instance returned by read_instances from which we want
        to build the vocabulary.
    vocab_size : ``int``
        Maximum size of vocabulary
    add_tokens : ``List[str]``
        if passed, those words will be added to vocabulary first.
    """
    print("\nBuilding Vocabulary.")

    # make sure pad_token is on index 0
    UNK_TOKEN = "@UNK@"
    PAD_TOKEN = "@PAD@"
    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    # First add tokens which were explicitly passed.
    add_tokens = add_tokens or []
    for token in add_tokens:
        if not token.lower() in token_to_id:
            token_to_id[token] = len(token_to_id)

    # Add remaining tokens from the instances as the space permits
    words = []
    for instance in instances:
        words.extend(instance["sentence1_tokens"] + instance["sentence2_tokens"])
    token_counts = dict(Counter(words).most_common(vocab_size))
    for token, _ in token_counts.items():
        token = token.strip()
        if not token:
            continue
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
        if len(token_to_id) == vocab_size:
            break
    # Make reverse vocabulary lookup
    id_to_token = dict(zip(token_to_id.values(), token_to_id.keys()))
    return token_to_id, id_to_token


def save_vocabulary(vocab_id_to_token: Dict[int, str], vocabulary_path: str) -> None:
    """
    Saves vocabulary to vocabulary_path.
    """
    with open(vocabulary_path, "w", encoding='utf-8') as file:
        # line number is the index of the token
        for idx in range(len(vocab_id_to_token)):
            file.write(vocab_id_to_token[idx] + "\n")
        file.close()


def load_vocabulary(vocabulary_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Loads vocabulary from vocabulary_path.
    """
    vocab_id_to_token = {}
    vocab_token_to_id = {}
    with open(vocabulary_path, "r", encoding='utf-8') as file:
        for index, token in enumerate(file):
            token = token.strip()
            if not token:
                continue
            vocab_id_to_token[index] = token
            vocab_token_to_id[token] = index
    return (vocab_token_to_id, vocab_id_to_token)


def load_glove_embeddings(embeddings_txt_file: str,
                          embedding_dim: int,
                          vocab_id_to_token: Dict[int, str]) -> np.ndarray:
    """
    Given a vocabulary (mapping from index to token), this function builds
    an embedding matrix of vocabulary size in which ith row vector is an
    entry from pretrained embeddings (loaded from embeddings_txt_file).
    """
    tokens_to_keep = set(vocab_id_to_token.values())
    vocab_size = len(vocab_id_to_token)

    embeddings = {}
    print("\nReading pretrained embedding file.")
    with open(embeddings_txt_file, encoding='utf-8') as file:
        for line in tqdm(file):
            line = str(line).strip()
            token = line.split(' ', 1)[0]
            if not token in tokens_to_keep:
                continue
            fields = line.rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                raise Exception(f"Pretrained embedding vector and expected "
                                f"embedding_dim do not match for {token}.")
                continue
            vector = np.asarray(fields[1:], dtype='float32')
            embeddings[token] = vector

    # Estimate mean and std variation in embeddings and initialize it random normally with it
    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))

    embedding_matrix = np.random.normal(embeddings_mean, embeddings_std,
                                        (vocab_size, embedding_dim))
    embedding_matrix = np.asarray(embedding_matrix, dtype='float32')
    for idx, token in vocab_id_to_token.items():
        if token in embeddings:
            embedding_matrix[idx] = embeddings[token]

    return embedding_matrix


def get_tokens(instance, tokenizer, device, glove_model, model_name, max_len):
    data1 = str(instance["sentence1"])
    data1 = " ".join(data1.split())

    data2 = str(instance["sentence2"])
    data2 = " ".join(data2.split())

    if model_name.startswith('bert'):
        inputs = tokenizer(
            data1,
            data2,
            add_special_tokens=True,
            max_length=min(max_len, 512),
            padding='max_length',
            truncation=True
            # return_tensors='pt'
        )
        instance['ids'] = inputs["input_ids"]
        instance['bert_mask'] = inputs['attention_mask']
        instance['token_type_ids'] = inputs["token_type_ids"]
    elif model_name.startswith('sbert'):
        inputs = tokenizer(
            [data1, data2],
            add_special_tokens=True,
            max_length=min(max_len, 512),
            padding='max_length',
            truncation=True
        )
        instance['data1_ids'] = inputs["input_ids"][0]
        instance['data1_bert_mask'] = inputs['attention_mask'][0]
        instance['data2_ids'] = inputs["input_ids"][1]
        instance['data2_bert_mask'] = inputs['attention_mask'][1]

    if glove_model:
        data1_token_ids = np.zeros(max_len)
        data2_token_ids = np.zeros(max_len)
        data1_token_ids[:len(instance['sentence1_token_ids'])] = instance['sentence1_token_ids']
        data2_token_ids[:len(instance['sentence2_token_ids'])] = instance['sentence2_token_ids']

        data1_mask = np.zeros(max_len)
        data2_mask = np.zeros(max_len)
        data1_mask[:len(instance['sentence1_token_ids'])] = 1
        data2_mask[:len(instance['sentence2_token_ids'])] = 1
        instance['data1_token_ids'] = data1_token_ids.tolist()
        instance['data2_token_ids'] = data2_token_ids.tolist()
        instance['data1_mask'] = data1_mask.tolist()
        instance['data2_mask'] = data2_mask.tolist()
    instance['targets'] = instance['label']
    return instance


def generate_batches(instances: List[Dict], batch_size, device, glove_model, model_name, pretrained_model_name):
    """
    Generates and returns batch of tensorized instances in a chunk of batch_size.
    """
    if model_name.startswith('bert'):
        tokenizer = AlbertTokenizer.from_pretrained(pretrained_model_name)
    elif model_name.startswith('sbert'):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    def chunk(items: List[Any], num: int):
        return [items[index:index+num] for index in range(0, len(items), num)]
    batches_of_instances = chunk(instances, batch_size)

    batches = []
    for batch_of_instances in tqdm(batches_of_instances):
        max_lens = [(len(instance["sentence1"]), len(instance["sentence2"])) for instance in batch_of_instances]
        max_data1 = max(max_lens, key=lambda x: x[0])[0]
        max_data2 = max(max_lens, key=lambda x: x[1])[1]
        max_tuple = max(max_lens, key=lambda x: x[0] + x[1])
        max_len = max_tuple[0] + max_tuple[1]
        if model_name.startswith('sbert'):
            max_len = max(max_data1, max_data2)
        for index in range(len(batch_of_instances)):
            batch_of_instances[index] = get_tokens(instance=batch_of_instances[index], tokenizer=tokenizer,
                                                   device=device, glove_model=glove_model, model_name=model_name,
                                                   max_len=max_len)
        batch_of_instances = pd.DataFrame(batch_of_instances)
        if model_name.startswith('bert'):
            bert_inputs = {
                'ids': torch.tensor(batch_of_instances["ids"], dtype=torch.long, device=device),
                'bert_mask': torch.tensor(batch_of_instances['bert_mask'], dtype=torch.long, device=device),
                'token_type_ids': torch.tensor(batch_of_instances["token_type_ids"], dtype=torch.long, device=device)
            }
        elif model_name.startswith('sbert'):
            bert_inputs = {
                'data1_ids': torch.tensor(batch_of_instances["data1_ids"], dtype=torch.long, device=device),
                'data1_bert_mask': torch.tensor(batch_of_instances['data1_bert_mask'], dtype=torch.long, device=device),
                'data2_ids': torch.tensor(batch_of_instances["data2_ids"], dtype=torch.long, device=device),
                'data2_bert_mask': torch.tensor(batch_of_instances['data2_bert_mask'], dtype=torch.long, device=device),
            }
        if glove_model:
            glove_inputs = {
                'data1_token_ids': torch.tensor(batch_of_instances['data1_token_ids'], dtype=torch.long, device=device),
                'data2_token_ids': torch.tensor(batch_of_instances['data2_token_ids'], dtype=torch.long, device=device),
                'data1_mask': torch.tensor(batch_of_instances['data1_mask'], device=device),
                'data2_mask': torch.tensor(batch_of_instances['data2_mask'], device=device)
            }
            batches.append({
                "bert_input": bert_inputs,
                "glove_inputs": glove_inputs,
                "targets": torch.tensor(batch_of_instances['targets'], device=device)
            })
        else:
            batches.append({
                "bert_input": bert_inputs,
                "targets": torch.tensor(batch_of_instances['targets'], device=device)
            })
    return batches


def save_ckp(state, checkpoint_dir='checkpoints'):
    f_path = checkpoint_dir + '/checkpoint.pt'
    torch.save(state, f_path)


def load_ckp(model, optimizer, scheduler, device, checkpoint_path='checkpoints/checkpoint.pt'):
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch']
