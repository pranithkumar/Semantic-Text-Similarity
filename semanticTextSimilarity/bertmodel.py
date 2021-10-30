import torch.nn as nn
import transformers


class BERTClassification(nn.Module):
    def __init__(self):
        super(BERTClassification, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-cased')
        self.bert_drop = nn.Dropout(0.4)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        d1, pooledOut = self.bert(ids, attention_mask=mask,
                                 token_type_ids=token_type_ids)
        bertOut = self.bert_drop(pooledOut)
        # print(d1)
        # print(pooledOut)
        output = self.out(bertOut)

        return output
