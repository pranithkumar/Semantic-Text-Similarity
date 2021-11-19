import pandas as pd
import torch
import transformers
from collections import Counter
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import numpy as np
import spacy
import json
import shutil
import random

seed_val = 1337

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

nlp = spacy.load("en_core_web_sm", disable=['ner', 'tagger', 'parser', 'textcat'])


class DATALoader:
    def __init__(self, data, max_length, glove_model=False):
        self.tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2')
        self.data1 = data.sentence1.values
        self.data2 = data.sentence2.values
        self.target = data.label.values
        self.max_length = max_length
        self.glove_model = glove_model
        if self.glove_model:
            self.data1_tokens = data.sentence1_token_ids.values
            self.data2_tokens = data.sentence2_token_ids.values
            self.data1_max_length = max([len(x) for x in self.data1_tokens])
            self.data2_max_length = max([len(x) for x in self.data2_tokens])
        else:
            self.data1_tokens, self.data2_tokens, self.data1_max_length, self.data2_max_length = None, None, None, None

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, item):
        data1 = str(self.data1[item])
        data1 = " ".join(data1.split())

        data2 = str(self.data2[item])
        data2 = " ".join(data2.split())

        inputs = self.tokenizer(
            data1,
            data2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
            # return_tensors='pt'
        )

        ids = inputs["input_ids"]
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        if not self.glove_model:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.target[item], dtype=torch.long)
            }

        data1_token_ids = np.zeros(self.data1_max_length)
        data2_token_ids = np.zeros(self.data2_max_length)
        data1_token_ids[:len(self.data1_tokens[item])] = self.data1_tokens[item]
        data2_token_ids[:len(self.data2_tokens[item])] = self.data2_tokens[item]

        data1_mask = np.zeros(self.data1_max_length)
        data2_mask = np.zeros(self.data2_max_length)
        data1_mask[:len(self.data1_tokens[item])] = 1
        data2_mask[:len(self.data2_tokens[item])] = 1

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.long),
            'data1_token_ids': torch.tensor(data1_token_ids, dtype=torch.long),
            'data2_token_ids': torch.tensor(data2_token_ids, dtype=torch.long),
            'data1_mask': torch.tensor(data1_mask),
            'data2_mask': torch.tensor(data2_mask)
        }

def read_instances(data_file_path: str,
                   max_allowed_num_tokens: int = 150) -> List[Dict]:
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
        # file.write("dummy\n")
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


def save_ckp(state, is_best: True, checkpoint_dir='checkpoints', best_model_dir='models'):
    f_path = checkpoint_dir + '/checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_path = best_model_dir + '/best_model.pt'
        shutil.copyfile(f_path, best_path)


def load_ckp(model, optimizer, scheduler, device, checkpoint_path='checkpoints/checkpoint.pt'):
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch']