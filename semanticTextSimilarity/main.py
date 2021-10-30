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
    df = pd.read_json('data/snli_1.0/snli_1.0_dev.jsonl', lines=True)
    df['contradiction'] = np.where(df['gold_label'] == 'contradiction', 1, 0)
    df['neutral'] = np.where(df['gold_label'] == 'neutral', 1, 0)
    df['entailment'] = np.where(df['gold_label'] == 'entailment', 1, 0)
    df['merged_labels'] = df[['contradiction', 'neutral', 'entailment']].values.tolist()
    df.drop(['contradiction', 'neutral', 'entailment'], axis=1)

    data = pd.DataFrame({
        'sentence1': df['sentence1'],
        'sentence2': df['sentence2'],
        'label': df['merged_labels']
    })

    # encoder = LabelEncoder()
    # data['label'] = encoder.fit_transform(data['label'])

    df_train, df_valid = train_test_split(data, test_size=0.1, random_state=23)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # df_train = df_train.head(100)

    train_dataset = DATALoader(
        data1=df_train.sentence1.values,
        data2=df_train.sentence2.values,
        target=df_train.label.values,
        max_length=512
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=4,
    )

    val_dataset = DATALoader(
        data1=df_valid.sentence1.values,
        data2=df_valid.sentence2.values,
        target=df_valid.label.values,
        max_length=512
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        num_workers=1,
    )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    print("Device: ",device)

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
        accuracy = metrics.accuracy_score(targets.argmax(1), outputs.argmax(1))
        print(f"Accuracy Score: {accuracy}")

        if accuracy > best_accuracy:
            torch.save(model.state_dict(), "models/model.bin")
            best_accuracy = accuracy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
