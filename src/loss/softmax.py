import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy

class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses, **kwargs):
		super(LossFunction, self).__init__()

		self.test_normalize = True
		
		self.criterion  = torch.nn.CrossEntropyLoss()
		self.fc 		= nn.Linear(nOut,nClasses)

		print('Initialised Softmax Loss')

	def forward(self, x, label=None):

		x 		= self.fc(x)
		nloss   = self.criterion(x, label)
		prec1	= accuracy(x.detach(), label.detach(), topk=(1,))[0]

		return nloss, prec1