import torch
import torch.nn as nn
from sklearn import metrics
from utils import prepare_dataloaders
from model import CompatModel

import argparse
parser = argparse.ArgumentParser(description='Fashion Compatibility Evaluation.')
parser.add_argument('--vse_off', action="store_true")
parser.add_argument('--pe_off', action="store_true")
parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--conv_feats', type=str, default="1234")
parser.add_argument('--model_path', type=str, default="./Final.pth")
args = parser.parse_args()

print(args)
vse_off = args.vse_off
pe_off = args.pe_off
mlp_layers = args.mlp_layers
conv_feats = args.conv_feats
model_path = args.model_path

train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
    prepare_dataloaders()
)

# Loading model
device = torch.device("cuda:0")
model = CompatModel(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary),
                    vse_off=vse_off, pe_off=pe_off, mlp_layers=mlp_layers, conv_feats=conv_feats).to(device)
model.load_state_dict(torch.load(model_path))
criterion = nn.BCELoss()

# Outfit Compatibility Test
model.eval()
total_loss = 0
outputs = []
targets = []
for batch_num, batch in enumerate(test_loader, 1):
    print("\r#{}/{}".format(batch_num, len(test_loader)), end="", flush=True)
    lengths, images, names, offsets, set_ids, labels, is_compat = batch
    images = images.to(device)
    target = is_compat.float().to(device)
    with torch.no_grad():
        output, _, _, _ = model._compute_score(images)
        output = output.squeeze(dim=1)
        loss = criterion(output, target)
    total_loss += loss.item()
    outputs.append(output)
    targets.append(target)
print()
print("Test Loss: {:.4f}".format(total_loss / batch_num))
outputs = torch.cat(outputs).cpu().data.numpy()
targets = torch.cat(targets).cpu().data.numpy()
print("AUC: {:.4f}".format(metrics.roc_auc_score(targets, outputs)))
