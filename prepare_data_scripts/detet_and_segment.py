from ultralytics import YOLO
from ultralytics import SAM
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
from pathlib import Path
from torchvision import transforms
import os
from PIL import Image
from ultralytics import solutions


#Uses cuda if nvidia gpu, xpu if intel gpu, and cpu if none are available
if torch.cuda.is_available():
    device = "cuda"
elif torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"
print("Using device:", device)    
#Loading the images from train_sample in datasets. If using own data
# change the text to the path of the dataset file.
image_paths = glob.glob("datasets/train_sample/*.jpg")
imgs = [cv2.imread(path) for path in image_paths]

#Setting model to Yolo26 (Newest as of now)
model = YOLO('yolo26n.pt')

#Has YOLO do object detection and make the bounding boxes
results = model.predict(imgs)


# Makes path for the cropped iamges to go to
cropped_directory = Path("datasets/crops")
cropped_directory.mkdir(exist_ok=True)
# Goes through each of the results from the YOLO model and gets the coordinates
# of the bounding boxes in order to crop the photos based on those bounding boxes
# cropped photos are then put in the CROPPED PHOTOS directory.
crop_list = []
for i, result in enumerate(results):
    img = cv2.imread(image_paths[i])
    if len(result) != 0:
        boxes = result.boxes.xyxy.tolist()
        # result.show() #(Uncomment if you want to see bounding boxes on the images)
        for x, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            #Adds a 20 pixel safety margin. Max and min stop crop from going over image size.
            height, width = img.shape[0], img.shape[1]
            crop = img[max(0, int(y1)-20):min(height, int(y2)+20), max(0, int(x1)-20):min(width,int(x2)+20)]
            crop_list.append(crop)
            # Ignores pictures with more than one bird in it
            if(len(boxes)<2):
                crop_name = Path(image_paths[i]).stem + f"_cropped-{x}.jpg"
                cv2.imwrite(str(cropped_directory / crop_name), crop)
            