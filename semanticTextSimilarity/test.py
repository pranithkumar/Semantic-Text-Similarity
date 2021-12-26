import torch
from tqdm import tqdm
import random
import numpy as np
seed_val = 1337

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def eval_func(data_loader, model, device, embedding_layer=None, glove_model=False):
    model.eval()

    fin_targets = []
    fin_output = []

    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):

            targets = d["targets"]
            targets = targets.to(device, dtype=torch.float)

            av1 = None
            av2 = None
            if glove_model:
                glove_data = d['glove_inputs']
                s1Embeds = embedding_layer(glove_data["data1_token_ids"])
                s2Embeds = embedding_layer(glove_data["data2_token_ids"])
                av1 = torch.sum(torch.mul(s1Embeds, glove_data['data1_mask'].unsqueeze(-1)), dim=1)
                av1 = torch.div(av1, torch.sum(glove_data['data1_mask'], dim=1).unsqueeze(-1))
                av2 = torch.sum(torch.mul(s2Embeds, glove_data['data2_mask'].unsqueeze(-1)), dim=1)
                av2 = torch.div(av2, torch.sum(glove_data['data2_mask'], dim=1).unsqueeze(-1))
            
            output = model(
                inputs=d['bert_input'],
                av1=av1,
                av2=av2
            )

            fin_targets.extend(targets.cpu().detach())
            fin_output.extend(output.cpu().detach())

        return torch.stack(fin_output), torch.stack(fin_targets)