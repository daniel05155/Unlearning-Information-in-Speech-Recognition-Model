# https://github.com/clam004/intro_continual_learning/tree/main
import argparse
from tqdm import tqdm
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio, torchvision
from torch.utils.data import DataLoader
from model import CNN
from dataloader import SpeechData, resample, AdvlossData, collate_fn_ver2, CustomSpeechData, Padding
from config import device, forget_class_num
from copy import deepcopy


##### Group data (Forget: 0, Retain: 1)  #####
def delete_row_tensor(pred, target, group, device):
    del_forget_row = [i for i in range(0, group.shape[0]) if group[i].item()==0]    
    del_retain_row = [j for j in range(0, group.shape[0]) if group[j].item()==1]
    
    ## Forget and retained prediction  ##
    pred_np = pred.cpu().detach().numpy()
    forget_np_pred = np.delete(pred_np, del_retain_row, 0)          # Deletion of retained part
    retain_np_pred = np.delete(pred_np, del_forget_row, 0)          # Deletion of forgotten part
    
    forget_pred = torch.from_numpy(forget_np_pred).to(device)
    retain_pred = torch.from_numpy(retain_np_pred).to(device)
    
    ## Forget and retained target  ##
    target_np = target.cpu().detach().numpy()
    forget_target_np = np.delete(target_np, del_retain_row, 0)      # Deletion of retained part
    retain_target_np = np.delete(target_np, del_forget_row, 0)      # Deletion of forgotten part
    
    forget_target = torch.from_numpy(forget_target_np).to(device)
    retain_target = torch.from_numpy(retain_target_np ).to(device)

    return forget_pred, forget_target, retain_pred, retain_target


## Make a copy of the old weights, ie theta_A,star, ie 𝜃∗A, in the loss equation  ##
def prev_param(model:nn.Module):
    _means = {}
    f_matrices = {}   # Create Fisher matrix
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in deepcopy(params).items():
        tmp = p.data
        _means[n] = p.data
        f_matrices[n] = tmp.zero_()
    return _means, f_matrices

## Fisher matrix and loss penality ## 
def _diag_fisher(model: nn.Module, theta_A, f_matrices, dataset):
    penality = 0
    for n, p in model.named_parameters():
        f_matrices[n].data += p.data ** 2 / len(dataset)
    for n, p in model.named_parameters():
        _val = f_matrices[n] * (p - theta_A[n]) ** 2
        penality += _val.sum()
    return penality


def unlearn(model, trainloader, criterion, optim, scheduler, num_epoch, batchsize, forget_class_num, alpha_set):
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load('/sppvenv/code/speech_cnn/checkpoint/model_train_best_ver1.pth', map_location=device))
    model.train()
    acc_list, loss_list = [], []
    # ckpt_name='./checkpoint/ver_0227_forget.pth'  # ckpt_name='./checkpoint/ver_' + epoch + '_0227_unlearn.pth'
    for alpha in alpha_set:
        ckpt_name='./checkpoint/ver_' + str(alpha) + '_0227_unlearn.pth'
        for epoch in range(0, num_epoch):
            unlearn_acc=0.0 
            unlearn_loss=0.0
            un_loss, re_loss=0.0, 0.0
            labelnum = 0                                    # Count labels
            for inputs, targets, group in tqdm(trainloader):
                inputs, targets, group = inputs.to(device), targets.to(device), group.to(device)
                group = torch.unsqueeze(group, dim=1)       # Part of dataset
                outputs = model(inputs).to(device)
                
                ## Model parameter - Theta_A and f_matrices ##
                theta_A, f_matrices = prev_param(model)    
                penality = _diag_fisher(model, theta_A, f_matrices, inputs)
                
                ## Take forget and retain prediction ## 
                forget_pred, forget_target, retain_pred, retain_target = delete_row_tensor(outputs, targets, group, device)
                
                ## Take forget and retain prediction ## 
                # forget_pred = (1-group) * outputs 
                # forget_targets = (1-group)*targets        
                # retain_pred = group * outputs  
                # retain_targets = group * targets
                
                ## Number of forget data in batch data is 0 ##
                if forget_pred.size(0) == 0:
                    re_loss = (criterion(retain_pred, retain_target))
                else:
                    un_loss = alpha*(criterion(retain_pred, retain_target)) + (criterion(forget_pred, forget_target)*(-1))
                optim.zero_grad()
                total_loss = re_loss+0.1*un_loss
                total_loss.requires_grad_(True)
                total_loss.backward()
                optim.step()
                scheduler.step()
                _, predicted = torch.max(outputs.data.squeeze(), dim=1)
                labelnum += targets.size(0)
                unlearn_acc += (predicted == targets).sum().item()
                unlearn_loss += total_loss.item()
            
            unlearn_acc = (unlearn_acc//labelnum)*100
            unlearn_loss = unlearn_loss/labelnum 
            acc_list.append(unlearn_acc)
            loss_list.append(unlearn_loss)
            print("Epoch: [{}], loss: {:.2f}, Acc: {:.2f}".format(epoch, unlearn_loss, unlearn_acc))
            torch.save(model.state_dict(), ckpt_name)
            print("========== Save Unlearning Model ==========")

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='Epoch', type=int, default=5)
    parser.add_argument('--batch_size', help='Batch Size', type=int, default=64)
    parser.add_argument('--partion_num', help='Number of each group', type=int, default=2000)
    parser.add_argument('--lr', help='Learning Rate', type=float, default=0.1)
    parser.add_argument('--alpha', help='Noise alpha', type=float, default=0.1)
    parser.add_argument('--retrain', help='Set retrained mode', action='store_true', default=False)
    args = parser.parse_args()
    
    # Parameter
    batch_size   = args.batch_size
    num_epoch    = args.epochs
    lr           = args.lr
    alpha        = args.alpha
    partion_num  = args.partion_num
    log_interval = 20
    sample_rate  = 16000
    new_sample_rate = 8000
    
    # Dataset  
    train_data = torchaudio.datasets.SPEECHCOMMANDS('/sppvenv/code/speech_cnn/data', download = True, subset='training')
    val_data   = torchaudio.datasets.SPEECHCOMMANDS('/sppvenv/code/speech_cnn/data', download = True, subset='validation')
    label_list = sorted(list(set(data[2] for data in val_data)))   # Number of label 
    transform = torchvision.transforms.Compose([Padding()]) 

    # train_set  = SpeechData(train_data, label_list)
    # forget_data, retain_data = resample(train_set, forget_class_num)
    # unlearning_data = AdvlossData(forget_data, retain_data)
    
    print("Preparing dataset...")
    unlearning_data = CustomSpeechData(train_data, label_list, transform, forget_class_num)
    # Dataloader
    train_loader = DataLoader(unlearning_data , batch_size=batch_size, shuffle=True, collate_fn=collate_fn_ver2, pin_memory=True)
    
    # Model
    model = CNN(num_class=len(label_list))
    model.to(device)
    
    # Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=num_epoch, steps_per_epoch=int(len(val_loader)),anneal_strategy='linear') 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Unlearning
    print("Go unlearn...")
    alpha_set = [0.3,0.5]
    unlearn(model, train_loader, criterion, optimizer, scheduler, num_epoch, batch_size, forget_class_num, alpha_set)
    
    print(" End Unlearning!!")
