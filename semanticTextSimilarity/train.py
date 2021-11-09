import torch
import torch.nn as nn
from tqdm import tqdm


def train_func(data_loader, model, optimizer, device, scheduler, embedding_layer):
    model.to(device)
    model.train()
    total_loss = 0
    count = 0
    enu = tqdm(enumerate(data_loader), total=len(data_loader))
    for bi, d in enu:
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]
        data1_token_ids = d["data1_token_ids"]
        data2_token_ids = d["data2_token_ids"]
        data1_mask = d['data1_mask']
        data2_mask = d['data2_mask']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        data1_token_ids = data1_token_ids.to(device, dtype=torch.long)
        data2_token_ids = data2_token_ids.to(device, dtype=torch.long)
        data1_mask = data1_mask.to(device, dtype=torch.long)
        data2_mask = data2_mask.to(device, dtype=torch.long)
        
        s1Embeds = embedding_layer(data1_token_ids)
        s2Embeds = embedding_layer(data2_token_ids)
        av1 = torch.sum(torch.mul(s1Embeds, data1_mask.unsqueeze(-1)), dim=1)
        av1 = torch.div(av1, torch.sum(data1_mask, dim=1).unsqueeze(-1))
        av2 = torch.sum(torch.mul(s2Embeds, data2_mask.unsqueeze(-1)), dim=1)
        av2 = torch.div(av2, torch.sum(data2_mask, dim=1).unsqueeze(-1))

        optimizer.zero_grad()
        output = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
            av=torch.cat((av1, av2), dim=1)
            # data1_token_ids=data1_token_ids,
            # data2_token_ids=data2_token_ids,
            # data1_mask=data1_mask,
            # data2_mask=data2_mask
        )

        loss = model.loss_fn(output, targets)
        total_loss += loss
        count += 1
        description = ("Average training loss: %.2f "
                           % (float(total_loss)/count))
        enu.set_description(description, refresh=False)
        loss.backward()

        optimizer.step()
        scheduler.step()


if __name__ == '__main__':
    # train_func()
    print("train")
