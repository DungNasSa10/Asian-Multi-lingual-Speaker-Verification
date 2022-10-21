#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse
import yaml
import torch
import glob
import warnings
from SpeakerNet import *
from utils import *
from DatasetLoader import *
from byol import *
warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',     type=int,   default=128,    help='Batch size, number of speakers per batch')
parser.add_argument('--nDataLoaderThread', type=int, default=4,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=1,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=100,    help='Maximum number of epochs')

# Model 
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.00005,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [lr_step] epochs')
parser.add_argument('--weight_decay',   type=float, default=2e-5,      help='Weight decay in the optimizer')
parser.add_argument('--lr_step',        type=int,   default=2,      help='Step for learning rate decay')
parser.add_argument('--step_size_up',   type=int,   default=20000,   help='step_size_up of CyclicLR')
parser.add_argument('--step_size_down',   type=int, default=20000,   help='step_size_down of CyclicLR')
parser.add_argument('--cyclic_mode',    type=str,   default='triangular2', help='policy of CyclicLR')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="data/MSV_CommonVoice_data/metadata/all_new_metadata2.txt",  help='Train list')
parser.add_argument('--test_list',      type=str,   default="data/Test/veri_test2.txt",   help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="/home2/vietvq/UNDERFITT/data/MSV_CommonVoice_data/", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="data/Test/wav", help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="data/musan_augment/", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="data/rirs_noises/", help='Absolute path to the test set')

# BYOL params
parser.add_argument('--projection_size',        type=int,    default=256)
parser.add_argument('--projection_hidden_size', type=int,    default=4096)
parser.add_argument('--moving_average_decay',   type=float,  default=0.99)
parser.add_argument('--use_momentum', dest='use_momentum', action='store_true')

## Distributed and mixed precision training
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

args = parser.parse_args()

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, args):

    args.gpu = gpu
    torch.backends.cudnn.benchmark = True

    ## Load models
    augment = AugmentWAV(musan_path=args.musan_path, rir_path=args.rir_path, max_frames=args.max_frames)
    speaker_net = SpeakerNet(args.model, "aamsoftmax", 1, nOut=192, nClasses=17714)
    wrapped_model = WrappedModel(speaker_net).cuda(0)
    self_learner = BYOL(speaker_net, augment, **vars(args))

    it = 1

    if args.gpu == 0:
        ## Write args to scorefile
        scorefile = open(args.result_save_path+"/scores.txt", "a+")

    ## Initialise trainer and data loader
    unlabeled_train_dataset = unlabeled_dataset_loader(**vars(args))

    train_loader = torch.utils.data.DataLoader(
        unlabeled_train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )

    trainer = ModelTrainer(wrapped_model, **vars(args))

    Optimizer = importlib.import_module("optimizer." + args.optimizer).__getattribute__("Optimizer")
    trainer.__optimizer__ = Optimizer(self_learner.parameters(), **vars(args))

    Scheduler = importlib.import_module("scheduler." + args.scheduler).__getattribute__("Scheduler")
    del args.optimizer
    trainer.__scheduler__, trainer.lr_step = Scheduler(trainer.__optimizer__, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    for ii in range(1, it):
        trainer.__scheduler__.step()

    ## Core training script
    for it in range(it, args.max_epoch + 1):
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}".format(it))

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        stepsize = train_loader.batch_size

        counter = 0
        index = 0
        loss = 0

        for data in train_loader:
            nloss = self_learner(data)
            trainer.__optimizer__.zero_grad()
            nloss.backward()
            trainer.__optimizer__.step()
            if args.use_momentum:
                self_learner.update_moving_average()

            loss += nloss.detach().cpu().item()
            counter += 1
            index += stepsize

            sys.stderr.write("Training {:d} / {:d}:".format(index, len(train_loader)*train_loader.batch_size))
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                                " Loss: %.5f, LR: %.7f \r" %(loss/counter, max([x['lr'] for x in trainer.__optimizer__.param_groups])))
            sys.stdout.flush()

            if trainer.lr_step == "iteration":
                trainer.__scheduler__.step()

        if trainer.lr_step == "epoch":
            trainer.__scheduler__.step()

        if args.gpu == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TLOSS {:f}, LR {:f}".format(it, loss, max(clr)))
            scorefile.write("Epoch {:d}, TLOSS {:f}, LR {:f} \n".format(it, loss, max(clr)))

        if it % args.test_interval == 0:

            sc, lab = trainer.eval_network(**vars(args))

            if args.gpu == 0:
                
                result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                print(time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}".format(it, result[1]), '\n')
                scorefile.write("Epoch {:d}, VEER {:2.4f}\n".format(it, result[1]))

                trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)

                scorefile.flush()

    if args.gpu == 0:
        scorefile.close()


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    main_worker(0, args)


if __name__ == '__main__':
    main()