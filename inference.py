import numpy as np
import torch
import torchaudio
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CNN
from dataloader import SpeechData
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataloader import SpeechData, collate_fn
from config import device, forget_class_num
import gc


def test(model, testloader, criterion, forget_class_num):
    model.load_state_dict(torch.load('./checkpoint/ver_0.5_0227_unlearn.pth', map_location=device))
    model.eval()
    test_loss, total, total_correct = 0, 0, 0
    y_pred, y_true = [], []
    with torch.no_grad():
        torch.cuda.empty_cache()
        for inputs, targets in tqdm(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # print(f"targets: {targets[0]}")
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            total_correct += (predicted == targets).sum().item()
            y_true.extend(targets.data.cpu().numpy())        # cf_matrix (Tensor -> Numpy)
            y_pred.extend(predicted.data.cpu().numpy()) 
    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_matrix = cf_matrix/np.sum(cf_matrix, axis=1)
    cf_matrix = np.around(cf_matrix, decimals=4)
    forget_acc, forget_num = 0.0, 0.0
    retain_acc, retain_num = 0.0, 0.0
    for idx in range(cf_matrix.shape[0]):
        print(f'cm[{idx}][{idx}]: {cf_matrix[idx][idx]},')
        if idx in forget_class_num:
            forget_num+=1
            forget_acc+=cf_matrix[idx][idx]
        else:
            retain_num+=1
            retain_acc+=cf_matrix[idx][idx]
    forget_acc=(forget_acc/forget_num)*100
    retain_acc=(retain_acc/retain_num)*100
    acc = 100. * total_correct / total
    print("Testing result.....\n")
    print("Loss: %.4f,  Overall accuracy: %.2f%%\n" %(test_loss/len(testloader), acc))
    print("Forget_acc: %.2f%%, Retain_acc: %.2f%%\n" %(forget_acc, retain_acc))


def main():
    # Parameter
    batch_size = 256
    sample_rate = 16000
    new_sample_rate = 8000
   
    test_data = torchaudio.datasets.SPEECHCOMMANDS('./data', download = True, subset='testing')     # len(test_data): 11005
    label_num = sorted(list(set(data[2] for data in test_data)))  # Classes of label                     
    test_set = SpeechData(test_data, label_num)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Model
    model = CNN(num_class=len(label_num))
    model.to(device)
    
    test(model, test_loader, criterion, forget_class_num)
    gc.collect()

if __name__ == '__main__':
    main()