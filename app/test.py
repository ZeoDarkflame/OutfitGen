
import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import random
import time
import os
import json
import sys
import torch
import re
from suggest import suggest

sys.path.insert(0, "../model")
import torchvision.transforms as transforms
from model import CompatModel
from utils import prepare_dataloaders
from PIL import Image

def get_options():
    data_root = "../data"
    img_root = os.path.join(data_root, "images")

    json_file = os.path.join(data_root, "test_no_dup_with_category_3more_name.json")
    json_file2 = os.path.join(data_root, "train_no_dup_with_category_3more_name.json")
    json_data2 = json.load(open(json_file2))
    print(len(json_data2.items()))

    json_data = json.load(open(json_file))
    json_data = {k:v for k, v in json_data.items() if os.path.exists(os.path.join(img_root, k))}

    print(len(json_data.items()))
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

    return top_options,bottom_options,shoe_options,bag_options,accessory_options

top_options,bottom_options,shoe_options,bag_options,accessory_options = get_options()