import torch


def Scheduler(optimizer, lr_decay, lr_step=1, **kwargs):

	sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)

	lr_step = 'epoch'

	print('Initialised step LR scheduler')

	return sche_fn, lr_step
