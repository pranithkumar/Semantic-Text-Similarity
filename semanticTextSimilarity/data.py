import torch
import transformers


class DATALoader:
    def __init__(self, data, target, max_length):
        self.data = data
        self.target = target  # make sure to convert the target into numerical values
        self.tokeniser = transformers.BertTokenizer.from_pretrained('bert-base-cased')
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = str(self.data[item])
        data = " ".join(data.split())

        inputs = self.tokeniser.encode_plus(
            data,
            None,
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


