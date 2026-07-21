import cv2, glob
sizes = [cv2.imread(p).shape[:2] for p in glob.glob("CROPPED_PHOTOS/*.jpg")]
print("median size:", sorted(h for h, w in sizes)[len(sizes)-1], sorted(w for h, w in sizes)[len(sizes)-1])