from torch.optim.lr_scheduler import CyclicLR


def Scheduler(optimizer, step_size_up=20000, step_size_down=20000, cyclic_mode='triangular2', **kwargs):

	sche_fn = CyclicLR(optimizer, base_lr=1e-5, max_lr=0.001, step_size_up=step_size_up, step_size_down=step_size_down, mode=cyclic_mode, cycle_momentum=False)

	lr_step = 'epoch'

	print('Initialised CyclicLR scheduler')

	return sche_fn, lr_step