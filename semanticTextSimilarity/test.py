import torch
from tqdm import tqdm


def eval_func(data_loader, model, device):
    model.eval()

    fin_targets = []
    fin_output = []

    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)


            output = model(
                ids=ids,
                masks = mask,
                token_type_ids = token_type_ids
            )

            fin_targets.extend(targets.cpu().detach().numpy().to_list())
            fin_targets.extend(torch.sigmoid(output).cpu().detach().numpy().to_list())

        return fin_output, fin_targets