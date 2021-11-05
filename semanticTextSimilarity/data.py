import pandas as pd
import torch
import transformers
from collections import Counter
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import numpy as np
import spacy, json


nlp = spacy.load("en_core_web_sm", disable=['ner', 'tagger', 'parser', 'textcat'])


class DATALoader:
    def __init__(self, data1, data2, target, max_length, data1_tokens, data2_tokens):
        self.data1 = data1
        self.data2 = data2
        self.target = target  # make sure to convert the target into numerical values
        self.tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2')
        self.max_length = max_length
        self.data1_tokens = data1_tokens
        self.data2_tokens = data2_tokens

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

        # padding_length = self.max_length - len(ids)
        # ids = ids + ([0] * padding_length)
        # mask = mask + ([0] * padding_length)
        # token_type_ids = token_type_ids + ([0] * padding_length)

        # ids = ids[0] + ids[1]
        # mask = mask[0] + mask[1]
        # token_type_ids = token_type_ids[0] + token_type_ids[1]

        data1_max_length = max([len(x) for x in self.data1_tokens])
        data2_max_length = max([len(x) for x in self.data2_tokens])
        data1_token_ids = np.array([np.pad(x, (0, data1_max_length - len(x)), 'constant', constant_values=-1) for x in self.data1_tokens])
        data2_token_ids = np.array([np.pad(x, (0, data2_max_length - len(x)), 'constant', constant_values=-1) for x in self.data2_tokens])

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.long),
            'data1_token_ids': torch.tensor(data1_token_ids, dtype=torch.long),
            'data2_token_ids': torch.tensor(data2_token_ids, dtype=torch.long)
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
    # instances = []
    # # instance = {'text': (data['sentence1'] + '' + data['sentence2']).tolist()}
    # for i in range(len(data)):
    #     instance = {'text': (data['sentence1'][i] + '' + data['sentence2'][i])}
    #     text = instance["text"]
    #     tokens = [token.text.lower() for token in nlp.tokenizer(text)][:max_allowed_num_tokens]
    #     # instance["labels"] = instance.pop("label", None)
    #     instance["text_tokens"] = tokens
    #     instance.pop("text")
    #     instances.append(instance)
    instances = []
    with open(data_file_path) as file:
        for line in tqdm(file.readlines()):
            instance = json.loads(line.strip())
            instance = {key: instance[key] for key in ["sentence1", "sentence2", "gold_label"]}
            tokens_sentence1 = [token.text.lower() for token in nlp.tokenizer(instance["sentence1"])][:max_allowed_num_tokens]
            tokens_sentence2 = [token.text.lower() for token in nlp.tokenizer(instance["sentence2"])][:max_allowed_num_tokens]
            # instance["labels"] = instance.pop("label", None)
            instance["sentence1_tokens"] = tokens_sentence1
            instance["sentence2_tokens"] = tokens_sentence2
            # instance.pop("sentence1")
            # instance.pop("sentence2")
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
                token_ids.append(1) # 1 is index for UNK
        instance["sentence1_token_ids"] = token_ids
        instance.pop("sentence1_tokens")
        token_ids = []
        for token in instance["sentence2_tokens"]:
            if token in token_to_id:
                token_ids.append(token_to_id[token])
            else:
                token_ids.append(1) # 1 is index for UNK
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
