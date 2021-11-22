import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AlbertModel
import random
import numpy as np
seed_val = 1337

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


class BERTClassification(nn.Module):
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def __init__(self, bert_trainable=True, glove_model=False, embedding_dim=0):
        super(BERTClassification, self).__init__()
        self.bert = AlbertModel.from_pretrained('albert-base-v2')
        self.bert_drop = nn.Dropout(0.1)
        self.glove_model = glove_model
        if glove_model:
            self.out = nn.Linear(768 + 2 * embedding_dim, 3)
        else:
            self.out = nn.Linear(768, 3)
        # self.cross_entropy = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self._init_weights(self.out)
        self._init_weights(self.bert)
        if not bert_trainable:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, inputs, av1=None, av2=None):
        _, pooledOut = self.bert(inputs['ids'], attention_mask=inputs['bert_mask'],
                                token_type_ids=inputs['token_type_ids'], return_dict=False)
        pooledOut = self.bert_drop(pooledOut)
        if self.glove_model:
            pooledOut = torch.cat((pooledOut, av1, av2), dim=1)
        output = self.out(pooledOut.float())

        return output

    def loss_fn(self, output, targets):
        # return self.cross_entropy(output, targets.argmax(1))
        return self.bce_loss(output, targets.float())


class SBERTClassification(nn.Module):
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def __init__(self, bert_trainable=True, glove_model=False, embedding_dim=0):
        super(SBERTClassification, self).__init__()
        self.bert = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.bert_drop = nn.Dropout(0.1)
        self.glove_model = glove_model
        if glove_model:
            self.out = nn.Linear(384 * 3 + 3 * embedding_dim, 3)
        else:
            self.out = nn.Linear(384 * 3, 3)
        # self.cross_entropy = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self._init_weights(self.out)
        self._init_weights(self.bert)
        if not bert_trainable:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, inputs, av1=None, av2=None):
        _, pooledOut1 = self.bert(inputs['data1_ids'], attention_mask=inputs['data1_bert_mask'],
                                  token_type_ids=inputs['data1_token_type_ids'], return_dict=False)
        _, pooledOut2 = self.bert(inputs['data2_ids'], attention_mask=inputs['data2_bert_mask'],
                                  token_type_ids=inputs['data2_token_type_ids'], return_dict=False)
        # bertOut = self.bert_drop(pooledOut)
        if self.glove_model:
            concat = torch.cat((pooledOut1, pooledOut1, torch.subtract(pooledOut1, pooledOut2), av1, av2, torch.subtract(av1, av2)), dim=1)
            output = self.out(concat.float())
        else:
            concat = torch.cat((pooledOut1, pooledOut1, torch.subtract(pooledOut1, pooledOut2)), dim=1)
            output = self.out(concat.float())

        return output

    def loss_fn(self, output, targets):
        # return self.cross_entropy(output, targets.argmax(1))
        return self.bce_loss(output, targets.float())
