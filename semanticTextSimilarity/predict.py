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
    df = pd.read_json('data/snli_1.0/snli_1.0_test.jsonl', lines=True)
    df['contradiction'] = np.where(df['gold_label'] == 'contradiction', 1, 0)
    df['neutral'] = np.where(df['gold_label'] == 'neutral', 1, 0)
    df['entailment'] = np.where(df['gold_label'] == 'entailment', 1, 0)
    df['merged_labels'] = df[['contradiction', 'neutral', 'entailment']].values.tolist()
    df.drop(['contradiction', 'neutral', 'entailment'], axis=1)

    df_test = pd.DataFrame({
        'sentence1': df['sentence1'],
        'sentence2': df['sentence2'],
        'label': df['merged_labels']
    })

    df_test = df_test.reset_index(drop=True)

    test_dataset = DATALoader(
        data1=df_test.sentence1.values,
        data2=df_test.sentence2.values,
        target=df_test.label.values,
        max_length=512
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        num_workers=4,
    )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    print("Device: ",device)

    model = BERTClassification()
    model.load_state_dict(torch.load("models/model_old.bin"))
    model.to(device)

    outputs, targets = eval_func(data_loader=test_data_loader, model=model, device=device)
    accuracy = metrics.accuracy_score(targets.argmax(1), outputs.argmax(1))
    print(f"Accuracy Score: {accuracy}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/