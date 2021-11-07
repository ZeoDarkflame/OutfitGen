import itertools
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class CategoryDataset(Dataset):
    
    def __init__(self,
                 root_dir="../data/images/",
                 data_file='train_no_dup_with_category_3more_name.json',
                 data_dir="../data",
                 transform=None,
                 use_mean_img=True,
                 neg_samples=True):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform
        self.use_mean_img = use_mean_img
        self.data = json.load(open(os.path.join(data_dir, data_file)))
        self.data = [(k, v) for k, v in self.data.items()]
        self.neg_samples = neg_samples
    
        self.vocabulary, self.word_to_idx = [], {}
        self.word_to_idx['UNK'] = len(self.word_to_idx)
        self.vocabulary.append('UNK')
        with open(os.path.join(self.data_dir, 'final_word_dict.txt')) as f:
            for line in f:
                name = line.strip().split()[0]
                if name not in self.word_to_idx:
                    self.word_to_idx[name] = len(self.word_to_idx)
                    self.vocabulary.append(name)


    def __getitem__(self, index):
        
        set_id, parts = self.data[index]
        if random.randint(0, 1) and self.neg_samples:
            to_change = list(parts.keys()) # randomly choose negative items
        else:
            to_change = []
        imgs = []
        labels = []
        names = []
        for part in ['upper', 'bottom', 'shoe', 'bag', 'accessory']:
            if part in to_change: # randomly choose a image from dataset with same category
                choice = self.data[index]
                while (choice[0] == set_id) or (part not in choice[1].keys()):
                    choice = random.choice(self.data)
                img_path = os.path.join(self.root_dir, str(choice[0]), str(choice[1][part]['index'])+'.jpg')
                names.append(torch.LongTensor(self.str_to_idx(choice[1][part]['name'])))
                labels.append('{}_{}'.format(choice[0], choice[1][part]['index']))
            elif part in parts.keys():
                img_path = os.path.join(self.root_dir, str(set_id), str(parts[part]['index'])+'.jpg')
                names.append(torch.LongTensor(self.str_to_idx(parts[part]['name'])))
                labels.append('{}_{}'.format(set_id, parts[part]['index']))
            elif self.use_mean_img:
                img_path = os.path.join(self.data_dir, part+'.png')
                names.append(torch.LongTensor([])) # mean_img embedding
                labels.append('{}_{}'.format(part, 'mean'))
            else:
                continue
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
        input_images = torch.stack(imgs)
        is_compat = (len(to_change)==0)

        offsets = list(itertools.accumulate([0] + [len(n) for n in names[:-1]]))
        offsets = torch.LongTensor(offsets)
        return input_images, names, offsets, set_id, labels, is_compat

    def __len__(self):
        return len(self.data)

    def str_to_idx(self, name):
        return [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx['UNK']
            for w in name.split()]

def collate_fn(data):
    """Need custom a collate_fn"""
    data.sort(key=lambda x:x[0].shape[0], reverse=True)
    images,  names, offsets, set_ids, labels, is_compat = zip(*data)
    lengths = [i.shape[0] for i in images]
    is_compat = torch.LongTensor(is_compat)
    names = sum(names, [])
    offsets = list(offsets)
    images = torch.stack(images)
    return (lengths,images,names,offsets,set_ids,labels,is_compat)


# testing loader
if __name__ == "__main__":
    img_size = 224
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor()
    ])
    d = CategoryDataset(transform=transform, use_mean_img=True)
    loader = DataLoader(d, 4, shuffle=True, num_workers=4, collate_fn=collate_fn)
