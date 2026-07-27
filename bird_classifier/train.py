# https://github.com/CV4EcologySchool/ct_classifier/blob/master/ct_classifier/train.py
import os
import argparse
import yaml
import glob
from tqdm import trange
from matplotlib import pyplot as plt
import numpy as np
import torch #
import torch.nn as nn 
from torch.utils.data import DataLoader 
from torch.optim import AdamW, SGD 
from util import init_seed
from dataset import BirdDataset
from model import EfficientNetModel
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import bitsandbytes as bnb
from pathlib import Path

#Builds a BirdDataset based on the split (training and validation) and wraps it
# in a pytorch dataloader.
def create_dataloader(cfg, split='train'):

    dataset_instance = BirdDataset(cfg, split)   

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers'],
            #from exp_efficientnet.yaml
            drop_last=(split == 'train')
        )
    return dataLoader


#Builds a new efficientnet model and loads the weights from the latest model state
# if there is one. Otherwise, starts from scratch
def load_model(cfg):

    model_instance = EfficientNetModel(cfg['num_classes'])     
    model_states = glob.glob('model_states/*.pt')
    if len(model_states):
        model_epochs = [int(m.replace('model_states/','').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch

    

#Saves the model weights into model_states
def save_model(cfg, epoch, model, stats):
    os.makedirs('model_states', exist_ok=True)

    stats['model'] = model.state_dict()

    torch.save(stats, open(f'model_states/{epoch}.pt', 'wb'))
    cfpath = 'model_states/config.yaml'
    #saves the configs if its the first time
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)


            
#Sets up the optimizer which adjusts the parameters to help the model learn
def setup_optimizer(cfg, model, freeze):


    #Changes to a lower learning rate once the backbone is unfrozen
    if (freeze == True):
        lr = cfg['learning_rate']
    else:
        lr = cfg['learning_rate_unfreeze']

    #deciding between Adam and SGD
    # I am using AdamW8bit because I don't have enough available GPU memory to run the normal AdamW 
    # If you have enough memory, feel free to use AdamW
    optimizer = bnb.optim.AdamW8bit(model.parameters(),
                    lr=lr ,
                    weight_decay=cfg['weight_decay']
                    # momentum=0.9, 
                    # nesterov=True
                    )
    print(lr)
    return optimizer

#Sets up cosine annealing which will adjust the learning rate. Helps prevent the loss and accuracy of the function from oscillating as much.
def setup_cosineannealing(cfg, optimizer):
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg['num_epochs']
        )
    return scheduler



def train(cfg, dataLoader, model, optimizer):
  
    device = cfg['device']
    model.to(device)

    model.train()
    #cross entropy loss function
    criterion = nn.CrossEntropyLoss()

    loss_total, oa_total = 0.0, 0.0

    progressBar = trange(len(dataLoader))
    for idx, (data, labels) in enumerate(dataLoader): 
        #loads onto device
        data, labels = data.to(device), labels.to(device)
        prediction = model(data)

        #Applies the optimizer step which updates the model's weights
        optimizer.zero_grad()
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()

        #Calculates the loss and overall accuracy to show and store
        loss_total += loss.item()
        pred_label = torch.argmax(prediction, dim=1)
        oa = torch.mean((pred_label == labels).float()) 
        oa_total += oa.item()

        #Bar to show progress
        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)
    progressBar.close()
    loss_total /= len(dataLoader) 
    oa_total /= len(dataLoader)
    if(device == 'xpu' and torch.xpu.is_available()):
        torch.xpu.empty_cache()
    return loss_total, oa_total
    


#Pretty much the same as the train function except this one sets the model to model.eval() other than model.train() and doesn't adjust weights
def validate(cfg, dataLoader, model):
  
    device = cfg['device']
    model.to(device)

    model.eval()
    
    criterion = nn.CrossEntropyLoss() 

    loss_total, oa_total = 0.0, 0.0   

    progressBar = trange(len(dataLoader))
    with torch.no_grad():
        for idx, (data, labels) in enumerate(dataLoader):

            data, labels = data.to(device), labels.to(device)

            prediction = model(data)
            loss = criterion(prediction, labels)
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    progressBar.close()
    if(device == 'xpu' and torch.xpu.is_available()):
        torch.xpu.empty_cache()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total

#Pretty much the same as val and this one also doesn't adjust the weights of the model
# and it's only run after training is finished.
def test(cfg, dataLoader, model):

    device = cfg['device']
    model.to(device)

    model.eval()

    criterion = nn.CrossEntropyLoss() 

    loss_total, oa_total = 0.0, 0.0   

    progressBar = trange(len(dataLoader))

    for idx, (data, labels) in enumerate(dataLoader): 
        #loads onto device
        data, labels = data.to(device), labels.to(device)
        prediction = model(data)

        loss = criterion(prediction, labels)
        loss.backward()

        loss_total += loss.item()
        pred_label = torch.argmax(prediction, dim=1)
        oa = torch.mean((pred_label == labels).float()) 
        oa_total += oa.item()

        progressBar.set_description(
            '[Test] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)
    progressBar.close()
    loss_total /= len(dataLoader) 
    oa_total /= len(dataLoader)

    return loss_total, oa_total

#Gets the accuracy information of the last saved model_state to graph later
def getOA():
    epochs = []
    paths = sorted(glob.glob(os.path.join("model_states", "*.pt")))
    for i, path in enumerate(paths):
        filename = os.path.basename(path)
        epochs_num = int(filename.split('.')[0])
        epochs.append(epochs_num)
    sorted_data = sorted(epochs) #Puts the info into a list of tuples and then sorts them based on the first value (epochs).
    path = Path(f"model_states/{sorted_data[-1]}.pt")
    
    print(path)
    oa_train = torch.load(path)['oa_train']
    oa_val = torch.load(path)['oa_val']
    return oa_train, oa_val

def main():

    # Loads the configurations like the seed and device
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_efficientnet.yaml')
    args = parser.parse_args()

    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    init_seed(cfg.get('seed', None))

    device = cfg['device']

        #I was crashing a lot so added this
    if(device == 'xpu' and torch.xpu.is_available()):
        torch.xpu.set_per_process_memory_fraction(0.7)
        
    if device != 'cpu' and not torch.xpu.is_available():
        print(f'WARNING: device set to "{device}" but XPU not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # makes the dataloaders
    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')
    dl_test = create_dataloader(cfg, split='test')

    #Loads model and optimizer
    model, current_epoch = load_model(cfg)
    #backbone frozen
    optim = setup_optimizer(cfg, model, True)
    scheduler = setup_cosineannealing(cfg, optim)

    #This does the training and validation for each epoch
    #Get epochs from confgis
    numEpochs = cfg['num_epochs']
    # Trains the model with a higher learning rate and the backbone frozen for the first 10 epochs in order to 
    # get the head of the model in a decent state.
    while current_epoch <= 10:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        #Training and validation
        loss_train, oa_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val = validate(cfg, dl_val, model)


        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val
        }
        save_model(cfg, current_epoch, model, stats)
        if(device == 'xpu' and torch.xpu.is_available()):
            torch.xpu.empty_cache()

    # reloads the optimizer wih the lower learning rate
    optim = setup_optimizer(cfg, model, False)
    #unfreezes the layers of the model 
    for param in model.feature_extractor.parameters():
        param.requires_grad = True
    print("Unfreezing and starting second phase")

    # Unfreezes the backbone of the model and starts training with a lower learning rate in order to let the rest
    # of the model train without ruining the weights from earlier.
    while current_epoch >10 and current_epoch<numEpochs:
        current_epoch += 1
        print(f'Epochs {current_epoch}/{numEpochs}')

        #Training and validation
        loss_train, oa_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val = validate(cfg, dl_val, model)
        scheduler.step()



        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val
        }
        save_model(cfg, current_epoch, model, stats)
        if(device == 'xpu' and torch.xpu.is_available()):
            torch.xpu.empty_cache()

    #Graphing the loss and accuracy for training, validation, and testing.
    oa_train, oa_val = getOA()
    loss_test, oa_test = test(cfg, dl_test, model)
    print("TEST LOSS: ", loss_test)
    print("TEST ACCURACY: ", oa_test)
    bars = plt.bar(["Training Accuracy", "Validation Accuracy", "Test Accuracy"],[oa_train*100, oa_val*100, oa_test*100])
    plt.title("Training, Validation, and Test Accuracy")
    plt.ylabel("Accuracy (%)")
    plt.ylim(top = 100)

    #https://stackoverflow.com/questions/53066633/how-to-show-values-on-top-of-bar-plot
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2, yval + 3, f"{round(yval, 2)}%", ha = "center")


    plt.show()



if __name__ == '__main__':

    main()