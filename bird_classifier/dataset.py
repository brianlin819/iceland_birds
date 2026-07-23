import os
import re
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, v2
from PIL import Image
import torch
import matplotlib.pyplot as plt

 
classes = ["ad_M", "ad_F", "juv_M", "juv_F"]
# Testing with only age instead of both at once
# classes = ["juv","ad"]
class_index = {}
for i, name in enumerate(classes):
    class_index[name] = i
 
class BirdDataset(Dataset):
 
    def __init__(self, config, split='train'):

        self.data_root = config['data_root']
        self.split = split
        #Doing the data augmentations for the training set but not validation or test set
        if split == 'train':
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(260),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomRotation((-45,45)),
                # v2.ColorJitter(brightness=0.1, contrast=0.1), #I don't know if I want to include this or not yet
 
            ])
            
        else:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(260),
            ])
        # path for annotations
        annoPath = os.path.join(
            self.data_root,
            'annotations',
            'training.json' if self.split == 'train' else ('val.json' if self.split =='val' else 'test.json')
        )
        meta = json.load(open(annoPath, 'r'))  
        labels = {}
        for entry in meta:
            age = entry.get("age")
            sex = entry.get("sex")
            if(age not in ("ad", "juv") or sex not in ("M", "F")): # skips annotations that don't have
                continue                                            # the age or sex in them.
            # if(age not in ("ad", "juv")): #skips the annotations that don't include 
            #     continue                  # the age of the bird in it.

            #Adding the filenames for the photos with annotations that include both the sex and the age.
            label = f"{age}_{sex}"
            # label = age
            stem = os.path.splitext(entry["filename"])[0]
            labels[stem] = label

        self.data = []
        image_dir = os.path.join(self.data_root, 'pads')
        for file in os.listdir(image_dir):
            stem = os.path.splitext(file)[0]
            # The two stems are to check for if the image has both the age and sex labels
            # It gets the filename of the photo in the same format as the annotations and then checks
            # with the list of annotations to see if the file is in there as well.
            if stem not in labels:
                continue
            self.data.append([file, class_index[labels[stem]]])

    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, index):

        image_name, label = self.data[index]
 
        image_path = os.path.join(self.data_root, 'pads', image_name)
        # I don't think I need this cause only cv2 changes the color but just in case.
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor, label
