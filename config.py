############# Global Setting #############  
import torch
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Target Label
forget_label=['backward', 'bird']

# Target Class
forget_class_num=[0]

