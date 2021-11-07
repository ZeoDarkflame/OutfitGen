import base64
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
import torch
from suggest import defect_detect,item_diagnosis,vec2mat,retrieve_sub,base64_to_tensor,base64_to_image

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
# loading model
model.load_state_dict(torch.load("../model/Final.pth", map_location="cpu"))
model.eval()
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False


json_file = os.path.join(data_root, "test_no_dup_with_category_3more_name.json")

json_data = json.load(open(json_file))
json_data = {k:v for k, v in json_data.items() if os.path.exists(os.path.join(img_root, k))}
len(json_data)

top_options, bottom_options, shoe_options, bag_options, accessory_options = [], [], [], [], []
print("Load options...")
for cnt, (iid, outfit) in enumerate(json_data.items()):
    if cnt > 10:
        break
    if "upper" in outfit:
        label = os.path.join(iid, str(outfit['upper']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        top_options.append({'label': label, 'value': value})
    if "bottom" in outfit:
        label = os.path.join(iid, str(outfit['bottom']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        bottom_options.append({'label': label, 'value': value})
    if "shoe" in outfit:
        label = os.path.join(iid, str(outfit['shoe']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        shoe_options.append({'label': label, 'value': value})
    if "bag" in outfit:
        label = os.path.join(iid, str(outfit['bag']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        bag_options.append({'label': label, 'value': value})
    if "accessory" in outfit:
        label = os.path.join(iid, str(outfit['accessory']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        accessory_options.append({'label': label, 'value': value})



print(len(top_options),len(bottom_options),len(shoe_options),len(bag_options),len(accessory_options))


top = base64.b64encode(open(top_options[5]['value'], "rb").read())
bottom = base64.b64encode(open(bottom_options[2]['value'], "rb").read())
shoe = base64.b64encode(open(shoe_options[0]['value'], "rb").read())
bag = base64.b64encode(open(bag_options[2]['value'], "rb").read())
accessory = base64.b64encode(open(accessory_options[1]['value'], "rb").read())
#print(top)

def outfit_plot():
    f, axarr = plt.subplots(1,5)
    axarr[0,0].imshow(base64_to_image(top.decode()))
    axarr[0,1].imshow(base64_to_image(bottom.decode()))
    axarr[0,2].imshow(base64_to_image(shoe.decode()))
    axarr[0,3].imshow(base64_to_image(bag.decode()))
    axarr[0,4].imshow(base64_to_image(accessory.decode()))
    plt.show()

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

outfit_plot()

for item in best_img_path.keys():
    print(item)
    print(best_img_path[item])
    if(item == 'upper'):
        top = base64.b64encode(open(best_img_path[item], "rb").read())
    elif(item == 'bottom'):
        bottom = base64.b64encode(open(best_img_path[item], "rb").read())
    elif(item == 'bag'):
        bag = base64.b64encode(open(best_img_path[item], "rb").read())
    elif(item == 'shoe'):
        shoe = base64.b64encode(open(best_img_path[item], "rb").read())
    elif(item == 'accessory'):
        accessory = base64.b64encode(open(best_img_path[item], "rb").read())
    
    outfit_plot()