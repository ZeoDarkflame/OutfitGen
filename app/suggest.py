import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import random
import os
import json
import sys
import torch
import re

sys.path.insert(0, "../model")
import torchvision.transforms as transforms
from model import CompatModel
from utils import prepare_dataloaders
from PIL import Image


data_root = "../data"
img_root = os.path.join(data_root, "images")

_, _, _, _, test_dataset, _ = prepare_dataloaders(root_dir=img_root, num_workers=1)
device = torch.device('cpu')
model = CompatModel(embed_size=1000, need_rep=True, vocabulary=2757).to(device)
# Loading model
model.load_state_dict(torch.load("../model/Final.pth", map_location="cpu"))
model.eval()
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

def defect_detect(img, model, normalize=True):
    # Register hook for comparison matrix
    relation = None

    def func_r(module, grad_in, grad_out):
        nonlocal relation
        relation = grad_in[1].detach()

    for name, module in model.named_modules():
        if name == 'predictor.0':
            module.register_backward_hook(func_r)
    # Forward
    out  = model._compute_score(img)
    out = out[0]

    # Backward
    one_hot = torch.FloatTensor([[-1]]).to(device)
    model.zero_grad()
    out.backward(gradient=one_hot, retain_graph=True)

    if normalize:
        relation = relation / (relation.max() - relation.min())
    relation += 1e-3
    return relation, out.item()

def item_diagnosis(relation, select):

    # finds the most incompatible item
    
    mats = vec2mat(relation, select)
    for m in mats:
        mask = torch.eye(*m.shape).byte()
        m.masked_fill_(mask, 0)
    result = torch.cat(mats).sum(dim=0)
    order = [i for i, j in sorted(enumerate(result), key=lambda x:x[1], reverse=True)]
    return result, order

def vec2mat(relation, select):
    
    mats = []
    for idx in range(4):
        mat = torch.zeros(5, 5)
        mat[np.triu_indices(5)] = relation[15*idx:15*(idx+1)]
        mat += torch.triu(mat, 1).transpose(0, 1)
        mat = mat[select, :]
        mat = mat[:, select]
        mats.append(mat)
    return mats

def retrieve_sub(x, select, order, try_most=5):
    
    # substitutes the worst item for the best choice

    all_names = {0:'upper', 1:'bottom', 2:'shoe', 3:'bag', 4:'accessory'}
   
    best_score = -1
    best_img_path = dict()

    for o in order:
        if best_score > 0.9:
            break
        problem_part_idx = select[o]
        problem_part = all_names[problem_part_idx]
        for outfit in random.sample(test_dataset.data, try_most):
            if best_score > 0.9:
                break
            if problem_part in outfit[1]:
                img_path = os.path.join(test_dataset.root_dir, outfit[0], str(outfit[1][problem_part]['index'])) + '.jpg'
                img = Image.open(img_path).convert('RGB')
                img = test_dataset.transform(img).to(device)
                x[0][problem_part_idx] = img
                with torch.no_grad():
                    out = model._compute_score(x)
                    score = out[0]
                if score.item() > best_score:
                    best_score = score.item()
                    best_img_path[problem_part] = img_path
        if problem_part in best_img_path:
            x[0][problem_part_idx] = test_dataset.transform(Image.open(best_img_path[problem_part]).convert('RGB')).to(device)
    
            print('problem_part: {}'.format(problem_part))
            print('best substitution: {} {}'.format(problem_part, best_img_path[problem_part]))
            print('After substitution the score is {:.4f}'.format(best_score))
    return best_score, best_img_path

def base64_to_tensor(image_bytes_dict):
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    outfit_tensor = []
    for k, v in image_bytes_dict.items():
        img = base64_to_image(v)
        tensor = my_transforms(img)
        outfit_tensor.append(tensor.squeeze())
    outfit_tensor = torch.stack(outfit_tensor)
    outfit_tensor = outfit_tensor.to(device)
    return outfit_tensor

def base64_to_image(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data).convert("RGB")
    return img


def suggest(type,path,top_options,bottom_options,shoe_options,bag_options,accessory_options):
    
    top_path = top_options[5]['value']
    bottom_path = bottom_options[2]['value']
    shoe_path = shoe_options[0]['value']
    bag_path = bag_options[2]['value']
    ac_path = accessory_options[1]['value']


    top = base64.b64encode(open(top_path, "rb").read())
    bottom = base64.b64encode(open(bottom_path, "rb").read())
    shoe = base64.b64encode(open(shoe_path, "rb").read())
    bag = base64.b64encode(open(bag_path, "rb").read())
    accessory = base64.b64encode(open(ac_path, "rb").read())

    print(path)
    if type == 'top':
        top = base64.b64encode(open(path, "rb").read())
        top_path = path
    elif type == 'bottom':
        bottom = base64.b64encode(open(path, "rb").read())
        bottom_path = path
    elif type == 'shoe':
        shoe = base64.b64encode(open(path, "rb").read())
        shoe_path = path
    elif type == 'bag':
        bag = base64.b64encode(open(path, "rb").read())
        bag_path = path
    elif type == 'accessory':
        accessory = base64.b64encode(open(path, "rb").read())
        ac_path = path
    
    img_dict = {
        "top": top.decode(),
        "bottom": bottom.decode(),
        "shoe": shoe.decode(),
        "bag": bag.decode(),
        "accessory": accessory.decode()
    }

    img_tensor = base64_to_tensor(img_dict)
    img_tensor.unsqueeze_(0)

    relation, score = defect_detect(img_tensor, model)
    print(score)
    relation = relation.squeeze()
    result, order = item_diagnosis(relation, select=[0, 1, 2, 3, 4])
    print(result,order)
    best_score, best_img_path = retrieve_sub(img_tensor, [0, 1, 2, 3, 4], order, 10)

    for item in best_img_path.keys():
        if(item == 'upper' and type != 'top'):
            top = base64.b64encode(open(best_img_path[item], "rb").read())
            top_path = best_img_path[item]
        elif(item == 'bottom' and type != 'bottom'):
            bottom = base64.b64encode(open(best_img_path[item], "rb").read())
            bottom_path = best_img_path[item]
        elif(item == 'bag' and type != 'bag'):
            bag = base64.b64encode(open(best_img_path[item], "rb").read())
            bag_path = best_img_path[item]
        elif(item == 'shoe' and type != 'shoe'):
            shoe = base64.b64encode(open(best_img_path[item], "rb").read())
            shoe_path = best_img_path[item]
        elif(item == 'accessory' and type != 'accessory'):
            accessory = base64.b64encode(open(best_img_path[item], "rb").read())
            ac_path = best_img_path[item]

    return top_path,bottom_path,shoe_path,bag_path,ac_path