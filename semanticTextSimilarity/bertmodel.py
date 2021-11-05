import torch
import torch.nn as nn
import transformers


class BERTClassification(nn.Module):
    def __init__(self):
        super(BERTClassification, self).__init__()
        self.bert = transformers.AlbertModel.from_pretrained('albert-base-v2')
        self.bert_drop = nn.Dropout(0.4)
        self.out = nn.Linear(768, 3)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, ids, mask, token_type_ids, data1_token_ids, data2_token_ids):
        # torch.cuda.empty_cache()
        with torch.no_grad():
            _, pooledOut = self.bert(ids, attention_mask=mask,
                             token_type_ids=token_type_ids, return_dict=False)
        bertOut = self.bert_drop(pooledOut)
        output = self.out(bertOut)

        return output

    def loss_fn(self, output, targets):
        return self.cross_entropy(output, targets.argmax(1))