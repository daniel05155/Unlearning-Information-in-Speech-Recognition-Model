from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import torch.utils.data
from config import  device

def var2device(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        # t = t.cuda()
        t = t.to(device)
    return Variable(t, **kwargs)

class EWC(object):  
    """
    Class to calculate the Fisher Information Matrix
    used in the Elastic Weight Consolidation portion
    of the loss function
    """
    def __init__(self, model: nn.Module, dataset: list):
        self.model = model      # pretrained model
        self.dataset = dataset  # samples from the old task or tasks

        # n is the string name of the parameter matrix p, aka theta, aka weights
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        
        # make a copy of the old weights, ie theta_A,star, ie 𝜃∗A, in the loss equation
        # we need this to calculate (𝜃 - 𝜃∗A)^2 because self.params will be changing 
        # upon every backward pass and parameter update by the optimizer
        self._means = {}
        for n, p in deepcopy(self.params).items():
            self._means[n] = var2device(p.data)
            
        # calculate the fisher information matrix 
        self._precision_matrices = self._diag_fisher()

    def _diag_fisher(self):
        # save a copy of the zero'd out version of
        # each layer's parameters of the same shape
        # to precision_matrices[n]
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = var2device(p.data)
        
        # we need the model to calculate the gradient but
        # we have no intention in this step to actually update the model
        # that will have to wait for the combining of this EWC loss term
        # with the new task's loss term
        self.model.eval()
        self.model.zero_grad()

        # remove channel dim, these are greyscale, not color rgb images
        # bs,1,h,w -> bs,h,w
        input, label = self.dataset
        input, label = input.to(device), label.to(device) 
        output = self.model(input)

        # calculate loss and backprop
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output,label)
        loss.backward()

        for n, p in self.model.named_parameters():
            precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
