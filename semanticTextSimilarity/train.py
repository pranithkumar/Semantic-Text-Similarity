import torch
import torch.nn as nn
from tqdm import tqdm
import random
import numpy as np

seed_val = 1337

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def train_func(data_loader, model, optimizer, device, scheduler, embedding_layer=None, glove_model=False):
    model.to(device)
    model.train()
    total_loss = 0
    count = 0
    enu = tqdm(enumerate(data_loader), total=len(data_loader))
    for bi, d in enu:

        targets = d["targets"]
        av1 = None
        av2 = None
        if glove_model:
            # We are fetching the glove embeddings of sentences and calculating the average vector for each sentence
            glove_data = d['glove_inputs']
            s1Embeds = embedding_layer(glove_data["data1_token_ids"])
            s2Embeds = embedding_layer(glove_data["data2_token_ids"])
            av1 = torch.sum(torch.mul(s1Embeds, glove_data['data1_mask'].unsqueeze(-1)), dim=1)
            av1 = torch.div(av1, torch.sum(glove_data['data1_mask'], dim=1).unsqueeze(-1))
            av2 = torch.sum(torch.mul(s2Embeds, glove_data['data2_mask'].unsqueeze(-1)), dim=1)
            av2 = torch.div(av2, torch.sum(glove_data['data2_mask'], dim=1).unsqueeze(-1))
        optimizer.zero_grad()
        output = model(
            inputs = d['bert_input'],
            av1=av1,
            av2=av2
        )

        loss = model.loss_fn(output, targets)
        total_loss += loss
        count += 1
        description = ("Average training loss: %.2f "
                       % (float(total_loss) / count))
        enu.set_description(description, refresh=False)

        loss.backward()

        optimizer.step()
        scheduler.step()
        