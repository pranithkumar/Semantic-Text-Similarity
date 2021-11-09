import torch
import torch.nn as nn
from transformers import AutoModel


class BERTClassification(nn.Module):
    def __init__(self, device, vocab_size, embedding_dim):
        super(BERTClassification, self).__init__()
        self.bert = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        # self.bert = transformers.AlbertModel.from_pretrained('albert-base-v2')
        self.bert_drop = nn.Dropout(0.4)
        # self.out = nn.Linear(768+2*embedding_dim, 3)
        self.out = nn.Linear(3*384, 3)
        self.cross_entropy = nn.CrossEntropyLoss()
        # self.embedding_layer = nn.Embedding(vocab_size, embedding_dim).to(device)

    def forward(self, data1_ids, data1_mask, data1_token_type_ids, data2_ids, data2_mask, data2_token_type_ids, av):
    #  token_type_ids, data1_token_ids, data2_token_ids, data1_mask, data2_mask):
        # torch.cuda.empty_cache()
        # s1Embeds = self.embedding_layer(data1_token_ids)
        # s2Embeds = self.embedding_layer(data2_token_ids)
        # av1 = torch.sum(torch.mul(s1Embeds, data1_mask.unsqueeze(-1)), dim=1)
        # av1 = torch.div(av1, torch.sum(data1_mask, dim=1).unsqueeze(-1))
        # av2 = torch.sum(torch.mul(s2Embeds, data2_mask.unsqueeze(-1)), dim=1)
        # av2 = torch.div(av2, torch.sum(data2_mask, dim=1).unsqueeze(-1))

        # with torch.no_grad():
        _, pooledOut1 = self.bert(data1_ids, attention_mask=data1_mask,
                             token_type_ids=data1_token_type_ids, return_dict=False)
        _, pooledOut2 = self.bert(data2_ids, attention_mask=data2_mask,
                             token_type_ids=data2_token_type_ids, return_dict=False)
        # bertOut = self.bert_drop(pooledOut)
        concat = torch.cat((pooledOut1, pooledOut1, torch.subtract(pooledOut1, pooledOut2)), dim=1)
        output = self.out(concat)

        return output

    def loss_fn(self, output, targets):
        return self.cross_entropy(output, targets.argmax(1))