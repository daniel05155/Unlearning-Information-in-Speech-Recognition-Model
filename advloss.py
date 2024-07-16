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
from dataloader import collate_fn, collate_fn_ver2, resample, resample_partion,AdvlossData, SpeechData
from unlearn_utils import UnLearningData
from torch.utils.data import DataLoader, Dataset
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def entropy_loss(x):
    out = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    out = -1.0 * out.sum(dim=1)
    return out.mean()

# def customloss(inputs , targets, gamma):
#     lossfn = nn.CrossEntropyLoss()
#     final_loss = (targets * lossfn) + targets * ( -(lossfn))
#     return final_loss

def forget(model, unlearnloader, num_epoch, optim, scheduler, model_save_path):
    print("Unlearning...")
    model.load_state_dict(torch.load('./checkpoint/model_best.pth', map_location=device))
    model.train()
    train_acc, train_loss, total = 0, 0, 0
    lossfn = nn.CrossEntropyLoss()
    for epoch in range(0, num_epoch):
        for (inputs, labels, group) in tqdm(unlearnloader):
            inputs, labels, group = inputs.to(device), labels.to(device), group.to(device)  # [batch_size, 1 , 16000]
            optim.zero_grad()
            outputs = model(inputs) # + 0.5*entropy_loss(inputs)
            # loss = group[0].item() * lossfn(outputs.squeeze(), labels) + ( (1 - group[0].item()) *(lossfn(outputs.squeeze(), labels)))
            if group[0].item()==0:
                # loss = lossfn(outputs.squeeze(), labels)
                loss = lossfn(outputs.squeeze(), labels) + 0.8*entropy_loss(inputs)
                (-loss).backward()
            else:
                loss = lossfn(outputs.squeeze(), labels)
                loss.backward()
            optim.step()
            scheduler.step()
            _, predicted = torch.max(outputs.data.squeeze(), dim=1)
            train_acc += (predicted == labels).sum().item()
            train_loss += loss.item() 
            total += labels.size(0)
        train_acc = train_acc*100/total
        train_loss = train_loss/len(unlearnloader)
    torch.save(model.state_dict(), os.path.join(model_save_path, 'adv_loss.pth'))
    print("========== Save Unlearning Model ==========")

def run():
    # Parameter
    num_epoch = 10
    log_interval = 20
    lr = 0.001
    # Forget Target Class
    forget_class = [0, 3]  

    # Dataset
    train_data = torchaudio.datasets.SPEECHCOMMANDS('./data', download = True, subset='training')
    val_data = torchaudio.datasets.SPEECHCOMMANDS('./data', download = True, subset='validation')
    # Count the number of each label
    label_num = sorted(list(set(data[2] for data in val_data)))  
    
    train_set = SpeechData(train_data, label_num)
    forget_data, retain_data = resample_partion(train_set, forget_class, len(train_set)/2)
    unlearning_data = AdvlossData(forget_data, retain_data)

    batch_size = len(forget_data) ### 
    unlearning_loader = DataLoader(unlearning_data, batch_size = batch_size, shuffle=False, collate_fn=collate_fn_ver2, pin_memory=True)
    
    # Model 
    model = CNN(num_class=len(label_num))  # model = NN2DMEL(num_class=len(label_num)) # Error 
    model.to(device)
    
    # Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=num_epoch, steps_per_epoch=int(len(unlearning_loader)),anneal_strategy='linear') 
    
    ##### Experiment Directory #####
    exp_name = model.__class__.__name__ + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
    exp_dir = os.path.join('./experiment-unlearn', exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    ##### Save Model Path #####
    model_save_path = os.path.join('./experiment-unlearn', exp_name, 'model')
    os.makedirs(model_save_path, exist_ok=True)
    
    ##### Config Directory #####
    log_dir = os.path.join(exp_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)

    ##### Write config #####
    config_path = os.path.join(log_dir, 'config_log.txt')
    write_config_log(config_path, model.__class__.__name__, num_epoch, batch_size, lr)

    # Unlearning
    forget(model, unlearning_loader, num_epoch, optimizer, scheduler, model_save_path)
    print("========== End Unlearning ==========")

if __name__ == '__main__':
    run()
