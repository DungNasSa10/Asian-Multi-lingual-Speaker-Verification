from torch.optim.lr_scheduler import CyclicLR


def Scheduler(optimizer, step_size_up, step_size_down, cyclic_mode, **kwargs):

	sche_fn = CyclicLR(optimizer, base_lr=0.00001, max_lr=0.01, step_size_up=step_size_up, step_size_down=step_size_down, mode=cyclic_mode)

	lr_step = 'epoch'

	print('Initialised CyclicLR scheduler')

	return sche_fn, lr_step