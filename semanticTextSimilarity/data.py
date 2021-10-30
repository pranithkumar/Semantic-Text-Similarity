import torch
import transformers


class DATALoader:
    def __init__(self, data1, data2, target, max_length):
        self.data1 = data1
        self.data2 = data2
        self.target = target  # make sure to convert the target into numerical values
        self.tokeniser = transformers.BertTokenizer.from_pretrained('bert-base-cased')
        self.max_length = max_length

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, item):
        data1 = str(self.data1[item])
        data1 = " ".join(data1.split())

        data2 = str(self.data2[item])
        data2 = " ".join(data2.split())

        inputs = self.tokeniser.encode_plus(
            data1,
            data2,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True

        )

        ids = inputs["input_ids"]
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        padding_length = self.max_length - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.long)
        }


