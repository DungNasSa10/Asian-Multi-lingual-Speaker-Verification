import torch


def Optimizer(parameters, lr=0.001, weight_decay=2e-5, **kwargs):

	print('Initialised Adam optimizer')

	return torch.optim.Adam(parameters, lr = lr, weight_decay = weight_decay)
