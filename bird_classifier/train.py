
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
from torch.optim import Adam, SGD 
from util import init_seed
from dataset import BirdDataset
from model import EfficientNetModel
import pandas as pd

torch.xpu.set_per_process_memory_fraction(1.0)
#Builds a BirdDataset based on the split (training and validation) and wraps it
# in a pytorch dataloader.
def create_dataloader(cfg, split='train'):

    dataset_instance = BirdDataset(cfg, split)   

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers']
            #from exp_efficientnet.yaml
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
def setup_optimizer(cfg, model):
    #deciding between Adam and SGD
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer



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

        optimizer.zero_grad()
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        pred_label = torch.argmax(prediction, dim=1)
        oa = torch.mean((pred_label == labels).float()) 
        oa_total += oa.item()

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

    return loss_total, oa_total



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
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total

def main():

    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_efficientnet.yaml')
    args = parser.parse_args()

    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    init_seed(cfg.get('seed', None))

    device = cfg['device']
    if device != 'cpu' and not torch.xpu.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')

    model, current_epoch = load_model(cfg)

    optim = setup_optimizer(cfg, model)

    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val = validate(cfg, dl_val, model)

        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val
        }
        save_model(cfg, current_epoch, model, stats)


if __name__ == '__main__':

    main()