import torch
import torch.nn as nn
import transformers
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
        self.bert = transformers.AlbertModel.from_pretrained('albert-base-v2')
        self.bert_drop = nn.Dropout(0.1)
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

    def forward(self, ids, mask, token_type_ids, av=None):
        _, pooledOut = self.bert(ids, attention_mask=mask,
                                token_type_ids=token_type_ids, return_dict=False)
        pooledOut = self.bert_drop(pooledOut)
        if av is not None:
            pooledOut = torch.cat((pooledOut, av), dim=1)
        output = self.out(pooledOut)

        return output

    def loss_fn(self, output, targets):
        # return self.cross_entropy(output, targets.argmax(1))
        return self.bce_loss(output, targets)