import os
from tqdm import tqdm
from datetime import datetime
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import CNN
from utils import write_config_log
from train_utils import train, val
from dataloader import SpeechData, collate_fn, resample
import argparse
from config import device, forget_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='Epoch', type=int, default=50)
    parser.add_argument('--batch_size', help='Batch Size', type=int, default=256)
    parser.add_argument('--lr', help='Learning Rate', type=float, default=0.001)
    parser.add_argument('--retrain', help='Set retrained mode', action='store_true', default=False)
    args = parser.parse_args()
    
    # Parameter
    batch_size = args.batch_size
    num_epoch = args.epochs
    lr        = args.lr
    log_interval = 20
    sample_rate = 16000
    new_sample_rate = 8000
    
    # Dataset  
    train_data = torchaudio.datasets.SPEECHCOMMANDS('./data', download = True, subset='training')
    val_data = torchaudio.datasets.SPEECHCOMMANDS('./data', download = True, subset='validation')
    
    # Number of label 
    label_num = sorted(list(set(data[2] for data in val_data)))
    
    # Retraining 
    if args.retrain:
        print('Resampling.....')
        forget_train_data, retain_train_data = resample(train_data, forget_label)     # Train Dataset
        forget_val_data, retain_val_data = resample(val_data, forget_label)           # Val Dataset
        train_set = SpeechData(retain_train_data, label_num)   
        val_set = SpeechData(retain_val_data, label_num)
    else:
        train_set = SpeechData(train_data, label_num)
        val_set = SpeechData(val_data, label_num)
    
    # Dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    
    # Model
    model = CNN(num_class=len(label_num))
    model.to(device)
    
    # Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=num_epoch, steps_per_epoch=int(len(train_loader)),anneal_strategy='linear') 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    ##### Experiment Directory #####
    exp_name = model.__class__.__name__ + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
    exp_dir = os.path.join('./experiment-train', exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    ##### Checkpoint Directory #####
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    
    ##### Save Model Path #####
    model_save_path = os.path.join('./experiment-train', exp_name, 'model')
    os.makedirs(model_save_path, exist_ok=True)
    
    ##### Config Directory #####
    log_dir = os.path.join(exp_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    ##### config / Training Process log #####
    config_path = os.path.join(log_dir, 'config_log.txt')
    result_log_path = os.path.join(log_dir, 'result_log.txt') 
    write_config_log(config_path, model.__class__.__name__, num_epoch, batch_size, lr)
    
    # Training
    for epoch in range(0, num_epoch):
        train(model, train_loader, criterion, optimizer, scheduler, epoch, device)
        best_acc = val(model, val_loader, criterion, epoch, num_epoch, device, result_log_path, model_save_path)
        print("best_acc:", best_acc)
    print("========== End Training ==========")

if __name__ == '__main__':
    main()
