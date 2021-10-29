import numpy as np
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


import torch
import torch.nn as nn
from tqdm import tqdm


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics


import pandas as pd

from bertmodel import BERTClassification
from data import DATALoader
from test import eval_func
from train import train_func

def run():
    df = pd.read_json('./data/snli_1.0_train.jsonl', lines=True)
    data = pd.DataFrame({
        'sentence1': df['sentence1'],
        'sentence2': df['sentence2'],
        'label': df['gold_label']
    })

    encoder = LabelEncoder()
    data['label'] = encoder.fit_transform(data['label'])

    df_train, df_valid = train_test_split(data, test_size=0.1, random_state=23, stratify=data.label.values)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = DATALoader(
        data=df_train.text.values,
        target=df_train.label.values,
        max_length=512
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=4,
    )

    val_dataset = DATALoader(
        data=df_valid.text.values,
        target=df_valid.label.values,
        max_length=512
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        num_workers=1,
    )

    device = torch.device("cuda")
    model = BERTClassification()

    param_optimizer = list(model.named_parameters())
    no_decay = [
        "bias",
        "LayerNorm,bias",
        "LayerNorm.weight",
    ]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    num_train_steps = int(len(df_train) / 8 * 10)

    optimizers = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizers,
        num_warmup_steps=0,
        num_training_steps=num_train_steps

    )

    best_accuracy = 0
    for epoch in range(5):
        train_func(data_loader=train_data_loader, model=model, optimizer=optimizers, device=device, scheduler=scheduler)
        outputs, targets = eval_func(data_loader=train_data_loader, model=model, device=device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score()
        print(f"Accuracy Score: {accuracy}")

        if accuracy > best_accuracy:
            torch.save(model.state_dict(), "model.bin")
            best_accuracy = accuracy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
