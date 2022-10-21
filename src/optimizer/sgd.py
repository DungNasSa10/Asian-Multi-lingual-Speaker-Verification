import torch


def Optimizer(parameters, lr=0.001, weight_decay=2e-5, **kwargs):

	print('Initialised SGD optimizer')

	return torch.optim.SGD(parameters, lr = lr, momentum = 0.9, weight_decay=weight_decay)
