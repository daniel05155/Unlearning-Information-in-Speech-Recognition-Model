import torch
import torchaudio
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import CNN
from train_utils import train, val
from unlearn_utils import unlearning_step, UnLearningData
from dataloader import SpeechData, collate_fn, resample
import argparse
from config import device

def blindspot_unlearner(model, unlearning_teacher, full_trained_teacher, forget_data, retain_data, 
                        epochs, lr , batch_size, KL_temperature, impaired_student_pth):
    # Creat the unlearning dataset.
    print("Re-label...")
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(unlearning_data, batch_size = batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    unlearning_teacher.eval()
    full_trained_teacher.eval()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    print("TS-Unlearning...")
    for epoch in tqdm(range(epochs)):  
        loss = unlearning_step(model = model, unlearning_teacher=unlearning_teacher, 
                        full_trained_teacher=full_trained_teacher, unlearn_data_loader=unlearning_loader, 
                        optimizer=optimizer, device=device, KL_temperature=KL_temperature, 
                        impaired_student_pth = impaired_student_pth)
        print("Epoch {} Unlearning Loss {}".format(epoch+1, loss))


def main():
    # Parameter
    batch_size = 256
    num_epoch = 2
    log_interval = 20
    lr = 0.001

    # Forget Target: 要改
    target = ['bird']  
    
    # Dataset
    train_data = torchaudio.datasets.SPEECHCOMMANDS('/sppvenv/code/speech_cnn/data', download=True, subset='training')
    val_data = torchaudio.datasets.SPEECHCOMMANDS('/sppvenv/code/speech_cnn/data', download=True, subset='validation')
    
    # Count the number of each label
    label_num = sorted(list(set(data[2] for data in val_data)))  

    # Data
    forget_train_data, retain_train_data = resample(train_data, target)
    
    # student Model
    student_model = CNN(num_class=len(label_num)).to(device)
    student_model.load_state_dict(torch.load("/sppvenv/code/speech_cnn/checkpoint/model_best.pth", map_location=device))

    # unlearned_teacher
    unlearning_teacher = CNN(num_class=len(label_num)).to(device) 
    # unlearning_teacher.load_state_dict(torch.load('./checkpoint/finetune_model.pth', map_location=device))  

    # full_trained_teacher
    prominent_teacher = CNN(num_class=len(label_num)).to(device) 
    prominent_teacher.load_state_dict(torch.load('/sppvenv/code/speech_cnn/checkpoint/finetune_model.pth', map_location=device))  
    prominent_teacher = prominent_teacher.eval()

    KL_temperature = 1
    blindspot_unlearner(model = student_model, unlearning_teacher = unlearning_teacher, full_trained_teacher = prominent_teacher, 
            forget_data = forget_train_data, retain_data = retain_train_data, epochs = num_epoch, lr = lr, 
            batch_size = batch_size, KL_temperature = KL_temperature, 
            impaired_student_pth = './checkpoint/After-TS_model_named_NN.pth')
    
if __name__ == '__main__':
    main()
