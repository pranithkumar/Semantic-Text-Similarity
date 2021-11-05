import torch
import torch.nn as nn
from tqdm import tqdm


def train_func(data_loader, model, optimizer, device, scheduler):
    model.to(device)
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]
        data1_token_ids = d["data1_token_ids"]
        data2_token_ids = d["data2_token_ids"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        data1_token_ids = data1_token_ids.to(device, dtype=torch.long)
        data2_token_ids = data2_token_ids.to(device, dtype=torch.long)

        optimizer.zero_grad()
        output = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
            data1_token_ids=data1_token_ids,
            data2_token_ids=data2_token_ids
        )

        loss = model.loss_fn(output, targets)
        loss.backward()

        optimizer.step()
        scheduler.step()


if __name__ == '__main__':
    # train_func()
    print("train")
