import cv2
import glob
from pathlib import Path
import random
import numpy as np

#Gets images from cropped photos folder
crop_paths = glob.glob("CROPPED_PHOTOS/*.jpg")
imgs = [cv2.imread(path) for path in crop_paths]

# Setting the desired pixel size of the photos and the
# directory to where the new photos will go
desired_size = 512
pad_directory = Path("READY_PHOTOS")
pad_directory.mkdir(exist_ok=True)

for i, img in enumerate(imgs):
    # I want to preserve the aspect ratio of the images while
    # scaling them down so I scale them based on the ratio of the 
    # desired size and the longest side of the image.
    h,w = img.shape[:2]
    if( h>w):
        ratio = desired_size/h
    else:
        ratio = desired_size/w
    # Getting the new pixel side lengths for the images
    new_h, new_w = round(h*ratio), round(w*ratio)
    # Testing to see if it worked
    print("OLD: ", h, w)
    print("NEW: ", new_h, new_w)
    # People online suggested using cv2.INTER_AREA for shrinking
    # images and cv2.INTER_CUBIC or cv2.INTER_LINEAR for enlargening
    if(ratio >1):
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC #Or us INTER_LINEAR
    #Resizing the image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation = interpolation)

    # Since the new image is not a perfect square, I am adding padding to the side that hasn't reached 512 pixels yet
    # to get it to a square

    # Getting how much padding I need for the sides
    remaining_h, remaining_w, = 512-new_h, 512-new_w
    # Separate the needed padding between top and bottom and right and left. Instead of just dividing by two, subtract from the remaining amount 
    # to avoid problems with odd numbers
    remaining_htop, remaining_wleft = round(remaining_h/2), round(remaining_w/2)
    remaining_bot, remaining_wright = remaining_h-remaining_htop, remaining_w-remaining_wleft

    #Adds the padding to the image and then saves it into the new directory.
    pad = cv2.copyMakeBorder(resized_img, remaining_htop, remaining_bot, remaining_wleft, remaining_wright, cv2.BORDER_CONSTANT)
    pad_name = Path(crop_paths[i]).stem + "_FINISHED.jpg"
    cv2.imwrite(str(pad_directory / pad_name), pad)


    #Data augmentation
    #Augmentation from here: https://www.kaggle.com/code/ahmedabdelfattah20/image-augmentation-using-opencv
    #Randomly flips the image horizontally and vertically
    augmented = cv2.flip(pad,random.randint(-1, 1)) 

    #Randomly rotates the image
    Cx , Cy = pad.shape[:2]
    rand_angle = random.randint(-45,45)
    M = cv2.getRotationMatrix2D((Cy//2, Cx//2),rand_angle ,1)
    augmented = cv2.warpAffine(augmented, M, (Cy, Cx))

    #Applies color jitters
    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(augmented)
    h += np.random.randint(0, 40,size=(Cx, Cy), dtype=np.uint8 )
    s += np.random.randint(0, 10,size=(Cx, Cy), dtype=np.uint8 )
    v += np.random.randint(0, 10,size=(Cx, Cy) , dtype=np.uint8 )
    augmented = cv2.merge([h,s,v ])
    augmented = cv2.cvtColor(augmented, cv2.COLOR_HSV2RGB)
    augmented_name = Path(crop_paths[i]).stem + "_FINISHED1.jpg"
    cv2.imwrite(str(pad_directory / augmented_name), augmented)
    


