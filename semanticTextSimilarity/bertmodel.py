import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AlbertModel, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
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

    def __init__(self, pretrained_model_name, model_name='bert', bert_trainable=True, glove_model=False,
                 embedding_dim=0):
        super(BERTClassification, self).__init__()
        self.bert = AlbertModel.from_pretrained(pretrained_model_name)
        self.model_name = model_name
        self.bert_drop = nn.Dropout(0.1)
        self.glove_model = glove_model
        if glove_model:
            self.out = nn.Linear(768 + 2 * embedding_dim, 3)
        else:
            self.out = nn.Linear(768, 3)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self._init_weights(self.out)
        self._init_weights(self.bert)
        if not bert_trainable:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, inputs, av1=None, av2=None):
        _, pooledOut = self.bert(inputs['ids'], attention_mask=inputs['bert_mask'],
                                 token_type_ids=inputs['token_type_ids'], 
                                 return_dict=False)
        pooledOut = self.bert_drop(pooledOut)
        if self.glove_model:
            pooledOut = torch.cat((pooledOut, av1), dim=1)
        output = self.out(pooledOut)

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

    def __init__(self, pretrained_model_name, model_name='sbert', bert_trainable=True, glove_model=False,
                 embedding_dim=0):
        super(SBERTClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.model_name = model_name
        self.bert_drop = nn.Dropout(0.1)
        self.glove_model = glove_model
        input_dim = 768 * 3
        if glove_model:
            input_dim += embedding_dim * 3
        if model_name == 'sbert_layer1':
            self.bert.config.output_hidden_states = True
            input_dim += 768 * 3

        self.out = nn.Linear(input_dim, 3, bias=False)
        self.cross_entropy = nn.CrossEntropyLoss()
        # self.mse_loss = nn.MSELoss()
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self._init_weights(self.out)
        self._init_weights(self.bert)
        if not bert_trainable:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, inputs, av1=None, av2=None):
        if self.model_name == 'sbert_cls':
            token_embeddings1, pooledOut1 = self.bert(inputs['data1_ids'], attention_mask=inputs['data1_bert_mask'],
                                                      # token_type_ids=inputs['data1_token_type_ids'],
                                                      return_dict=False)
            token_embeddings2, pooledOut2 = self.bert(inputs['data2_ids'], attention_mask=inputs['data2_bert_mask'],
                                                      # token_type_ids=inputs['data2_token_type_ids'],
                                                      return_dict=False)
        elif self.model_name == 'sbert_layer1':
            token_embeddings1, _, layer_output1 = self.bert(inputs['data1_ids'],
                                                            attention_mask=inputs['data1_bert_mask'],
                                                            # token_type_ids=inputs['data1_token_type_ids'],
                                                            return_dict=False)
            token_embeddings2, _, layer_output2 = self.bert(inputs['data2_ids'],
                                                            attention_mask=inputs['data2_bert_mask'],
                                                            # token_type_ids=inputs['data2_token_type_ids'],
                                                            return_dict=False)
            token_embeddings1 = self.mean_pooling(token_embeddings1, inputs['data1_bert_mask'])
            token_embeddings2 = self.mean_pooling(token_embeddings2, inputs['data2_bert_mask'])
            pooledOut1 = self.mean_pooling(layer_output1[1], inputs['data1_bert_mask'])
            pooledOut2 = self.mean_pooling(layer_output2[1], inputs['data2_bert_mask'])
            pooledOut1 = torch.concat((token_embeddings1, pooledOut1), dim=1)
            pooledOut2 = torch.concat((token_embeddings2, pooledOut2), dim=1)
        elif self.model_name == 'sbert':
            token_embeddings1, _ = self.bert(inputs['data1_ids'], attention_mask=inputs['data1_bert_mask'],
                                             # token_type_ids=inputs['data1_token_type_ids'],
                                             return_dict=False)
            token_embeddings2, _ = self.bert(inputs['data2_ids'], attention_mask=inputs['data2_bert_mask'],
                                             # token_type_ids=inputs['data2_token_type_ids'],
                                             return_dict=False)
            pooledOut1 = self.mean_pooling(token_embeddings1, inputs['data1_bert_mask'])
            pooledOut2 = self.mean_pooling(token_embeddings2, inputs['data2_bert_mask'])
        # pooledOut1 = self.bert_drop(pooledOut1)
        # pooledOut2 = self.bert_drop(pooledOut2)

        if self.glove_model:
            concat = torch.cat((pooledOut1, pooledOut2, torch.abs(torch.subtract(pooledOut1, pooledOut2)), av1, av2,
                                torch.abs(torch.subtract(av1, av2))), dim=1)
            output = self.out(concat.float())
        else:
            concat = torch.cat((pooledOut1, pooledOut2, torch.abs(torch.subtract(pooledOut1, pooledOut2))), dim=1)
            output = self.out(concat.float())
        return output

    def mean_pooling(self, token_embeddings, attention_mask):
        # token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def loss_fn(self, output, targets):
        # loss = self.mse_loss(output, targets.view(-1).float())
        # return loss.float()
        return self.cross_entropy(output, targets.argmax(1))
        # return self.bce_loss(output, targets.float())
