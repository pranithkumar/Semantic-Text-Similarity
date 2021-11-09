import torch
from tqdm import tqdm


def eval_func(data_loader, model, device, embedding_layer):
    model.eval()

    fin_targets = []
    fin_output = []

    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            data1_ids = d["data1_ids"]
            data1_token_type_ids = d["data1_token_type_ids"]
            data1_bert_mask = d["data1_bert_mask"]
            data2_ids = d["data2_ids"]
            data2_token_type_ids = d["data2_token_type_ids"]
            data2_bert_mask = d["data2_bert_mask"]
            targets = d["targets"]
            data1_token_ids = d["data1_token_ids"]
            data2_token_ids = d["data2_token_ids"]
            data1_mask = d['data1_mask']
            data2_mask = d['data2_mask']

            data1_ids = data1_ids.to(device, dtype=torch.long)
            data1_token_type_ids = data1_token_type_ids.to(device, dtype=torch.long)
            data1_bert_mask = data1_bert_mask.to(device, dtype=torch.long)
            data2_ids = data2_ids.to(device, dtype=torch.long)
            data2_token_type_ids = data2_token_type_ids.to(device, dtype=torch.long)
            data2_bert_mask = data2_bert_mask.to(device, dtype=torch.long)
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

            output = model(
                data1_ids=data1_ids,
                data1_mask=data1_bert_mask,
                data1_token_type_ids=data1_token_type_ids,
                data2_ids=data2_ids,
                data2_mask=data2_bert_mask,
                data2_token_type_ids=data2_token_type_ids,
                av=torch.cat((av1, av2), dim=1)
                # data1_token_ids=data1_token_ids,
                # data2_token_ids=data2_token_ids,
                # data1_mask=data1_mask,
                # data2_mask=data2_mask
            )

            fin_targets.extend(targets.cpu().detach())
            fin_output.extend(torch.sigmoid(output).cpu().detach())

        return torch.stack(fin_output), torch.stack(fin_targets)