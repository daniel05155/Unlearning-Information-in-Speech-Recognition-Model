import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset


## Create for TS ## 
class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)
    
    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, index):
        if(index < self.forget_len):
            x = self.forget_data[index][0]
            y = 1
            return x,y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x,y

##  Unlearning tool ## 
def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim = 1)
    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)
    
    # label 1 means forget sample, label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)

def unlearning_step(model, unlearning_teacher, full_trained_teacher, unlearn_data_loader, optimizer, 
                    device, KL_temperature, impaired_student_pth):
    losses = []
    for batch in unlearn_data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(output = output, labels=y, full_teacher_logits=full_teacher_logits, 
                unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        torch.save(model.state_dict(), impaired_student_pth)    # Save Unlearned Model
    return np.mean(losses) 
