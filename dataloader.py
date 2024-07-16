from tqdm import tqdm
import torch
import torchaudio
from torch.utils.data import Dataset
import torchvision
from config import forget_class_num


class SpeechData(Dataset):
    def __init__(self, data, label_dict, transform=None):
        self.data = data
        self.label_dict = label_dict
        self.transform = transform

    def __len__(self):
        return len(self.data) 

    def __getitem__(self,idx):
        waveform = self.data[idx][0]
        label = self.data[idx][2]
        if label in self.label_dict:
            out_labels = self.label_dict.index(label)            
        return waveform, out_labels

# zero pad to have 1 sec len
class Padding:
    def __init__(self):
        self.output_len = 16000  # Sample rate

    def __call__(self, x):
        pad_len = self.output_len - x.shape[-1]
        if pad_len > 0:
            x = torch.cat([x, torch.zeros([x.shape[0], pad_len])], dim=-1)
        elif pad_len < 0:
            raise ValueError("no sample exceed 1sec in GSC.")
        return x

##### Group data (Forget: 0, Retain: 1)  #####
class CustomSpeechData(Dataset):
    def __init__(self, data, label_list, transform, forget_list):
        self.data = data
        self.label_list = label_list
        self.transform = transform          # Padding
        self.forget_list = forget_list
        
    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        waveform = self.data[idx][0]
        waveform = self.transform(waveform)
        label = self.data[idx][2]
        out_labels = self.label_list.index(label)
        if out_labels in self.forget_list:
            group = 0.0
        else:
            group = 1.0
        return waveform, out_labels, group
        
##### For adversarial loss  #####
class AdvlossData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.forget_len + self.retain_len 
    
    def __getitem__(self, index):
        if(index < self.forget_len):
            x = self.forget_data[index][0]
            y = self.forget_data[index][1]
            group = 0.0
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = self.retain_data[index - self.forget_len][1]
            group = 1.0
        return x, y, group


##### Make all tensor in a batch the same length by padding with zeros  #####
def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

##### For TS  #####
def collate_fn(batch):
    # The form of data tuple: waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    for waveform, label in batch:
        tensors += [waveform]
        targets += [torch.tensor(label)]
    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets

##### For advloss #####
def collate_fn_ver2(batch):
    # Form of data tuple: waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets, groups = [], [], []
    # Gather in lists, and encode labels as indices
    for waveform, label, group in batch:
        tensors += [waveform]
        targets += [torch.tensor(label)]
        groups += [torch.tensor(group)]
    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    groups = torch.stack(groups)
    return tensors, targets, groups

##### Resampling #####
def resample(dataset, target):
    forget_data = []
    retain_data = []
    print("Resampling Dataset....")
    for idx in tqdm(range(len(dataset))):
        if dataset[idx][1] in target:
            forget_data.append(dataset[idx])
        else:
            retain_data.append(dataset[idx])
    return forget_data, retain_data

##  Forget and Retrained 
def resample_partion(dataset, target, partion_num):
    forget_data, retain_data = [], []
    print("Begin resampling...")
    for idx in tqdm(range(len(dataset))):
        if idx <= partion_num:
            if dataset[idx][1] in target:
                forget_data.append(dataset[idx])
            else:
                retain_data.append(dataset[idx])
        else:
            print("Finish Resampling...")
            break
    return forget_data, retain_data

# 先分群，再各群抓partion_num個
def group_data(dataset, partion_num, label_num):
    after_grouping = []
    group={}
    for i in range(label_num):
        group[i]=[]
    # 分類
    print("Grouping....")
    for idx in tqdm(range(len(dataset))):
        data, label = dataset[idx]
        group[label].append((data, label))
    # 每類別各抓partion_num個
    print("Take from each class...")
    for j in range(label_num):
        for k in range(partion_num):    
            after_grouping.append(group[j][k])
    print("Finsh grouping....")
    return after_grouping

if __name__ =='__main__':
    train_data = torchaudio.datasets.SPEECHCOMMANDS('/sppvenv/code/speech_cnn/data', download = True, subset='training')
    val_data = torchaudio.datasets.SPEECHCOMMANDS('/sppvenv/code/speech_cnn/data', download = True, subset='validation')
    label_list = sorted(list(set(data[2] for data in val_data)))

    # train_dataset = SpeechData(train_data, label_list)
    transform = torchvision.transforms.Compose([Padding()])
    train_dataset = CustomSpeechData(train_data, label_list, transform, forget_class_num)
    train_dataset  = torch.utils.data.DataLoader(dataset=train_dataset , shuffle=True, batch_size=32)
    
    # answer = group_data(train_dataset, partion_num=50, label_num=len(label_list))
    # print(len(answer))
