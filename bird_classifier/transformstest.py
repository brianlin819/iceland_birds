import cv2
import glob
from pathlib import Path
import random
import numpy as np
from torchvision.transforms import v2
import torch
import matplotlib.pyplot as plt
from PIL import Image

crop_paths = glob.glob("CROPPED_PHOTOS/*.jpg")
imgs = [cv2.imread(path) for path in crop_paths]

test_directory = Path("datasets/TEST_PHOTOS")
test_directory.mkdir(exist_ok=True)

for i, img in enumerate(imgs):

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transforms = v2.Compose([
        v2.ToImage(),
        v2.CenterCrop(528),
        v2.Resize(528),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
    ])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    aug_img = transforms(image_rgb)
    axes[1].imshow(aug_img.permute(1, 2, 0))
    axes[1].set_title("After Augmentation")
    axes[1].axis("off")
    plt.show()
