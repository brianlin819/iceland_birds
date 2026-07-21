import torch
from matplotlib import pyplot as plt
import glob
import os

paths = glob.glob(os.path.join("model_states", "*.pt"))

epochs, train_loss, val_loss, train_oa, val_oa = [], [], [],[],[]

for i, path in enumerate(paths):
    filename = os.path.basename(path)
    epochs_num = int(filename.split('.')[0])
    stats = torch.load(path)
    epochs.append(epochs_num)
    train_loss.append(stats['loss_train'])
    val_loss.append(stats['loss_val'])
    train_oa.append(stats['oa_train'])
    val_oa.append(stats['oa_val'])

sorted_data = sorted(zip(epochs, train_loss, val_loss, train_oa, val_oa)) #Generated from gemini
epochs, train_loss, val_loss, train_oa, val_oa = zip(*sorted_data)


print("TRAIN: ", val_oa)
print("VAL", train_oa)
plt.plot(train_loss, label = 'Train Loss', marker='o')
plt.plot(val_loss, label = 'Validation Loss', marker = 'o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.show()
#Overall Accuracy Plot
plt.plot(train_oa, label = "Training Accuracy", marker = 'o', color = 'orange')
plt.plot(val_oa, label = "Validation Accuracy", marker = 'o', color = 'green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
plt.title("Training Accuracy vs Validation Accuracy")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.show()  