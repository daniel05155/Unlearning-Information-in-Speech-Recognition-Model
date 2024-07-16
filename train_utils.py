import os
from tqdm import tqdm
import torch
from utils import write_result_log
import wandb

best_acc =0

def train(model, trainloader, criterion, optim, scheduler, epoch, device):
    print("Training...")
    model.train()
    train_acc, train_loss = 0, 0
    total = 0
    for inputs, targets in tqdm(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)  # [batch_size, 1 , 16000]
        optim.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)    # F.nll_loss(outputs.squeeze(), targets)
        loss.backward()
        optim.step()
        scheduler.step()
        _, predicted = torch.max(outputs.data.squeeze(), dim=1)
        train_acc += (predicted == targets).sum().item()
        train_loss += loss.item() 
        total += targets.size(0)
    train_acc = train_acc/total *100
    train_loss = train_loss/len(trainloader)
    # wandb.log({"train_acc": train_acc, "train_loss":train_loss})    
    print("Epoch: [{}], Train_loss: {:.2f}, Train_Acc: {:.2f}".format(epoch, train_loss, train_acc))

def val(model, val_loader, criterion, epoch, total_epoch, device, result_log_path, model_save_path):
    global best_acc
    model.eval()
    val_acc, val_loss, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data.squeeze(), 1)
            val_acc += (predicted == targets).sum().item()
            total += targets.size(0)
    
    val_acc = (val_acc/total) *100
    val_loss /= len(val_loader)
    # wandb.log({"val_acc": val_acc, "val_loss":val_loss})        
    print("\nValidation Epoch #[{}], Val Loss:{:.4f}, Val Acc:{:.4f}".format(epoch, val_loss, val_acc))
    # Save checkpoint with better result
    if val_acc > best_acc:
        print(f".....Saving Model.....{model_save_path}")
        torch.save(model.state_dict(), os.path.join(model_save_path, 'model_best.pth'))
        write_result_log(result_log_path, epoch, total_epoch, val_acc, val_loss, is_better=True)
        best_acc = val_acc
    return best_acc
