import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy


class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.35, scale=64, **kwargs):
        super(LossFunction, self).__init__()
        self.w = torch.nn.Parameter(torch.FloatTensor(nOut, nClasses), requires_grad=True)
        self.nClasses = nClasses
        self.m = margin
        self.s = scale
        nn.init.xavier_normal_(self.w, gain=1)

    def forward(self, x, labels):
        x_norm = x / torch.norm(x, dim=1, keepdim=True)
        w_norm = self.w / torch.norm(self.w, dim=0, keepdim=True)
        xw_norm = torch.matmul(x_norm, w_norm)

        label_one_hot = F.one_hot(labels.view(-1), self.nClasses).float() * self.m
        value = self.s * (xw_norm - label_one_hot)

        loss = F.cross_entropy(input=value, target=labels.view(-1))
        prec1 = accuracy(value.detach(), labels.view(-1).detach(), topk=(1,))[0]
        return loss, prec1