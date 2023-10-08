#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
import os
import json
import logging
from collections import OrderedDict
from contextlib import suppress


import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as NativeDDP
torch.multiprocessing.set_sharing_strategy('file_system')
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import  model_parameters
from timm.utils import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from My_Mobilenet3 import MobileNetV3

from prune_helper import prune_lowindex, calibrate

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')


# Model parameters
parser.add_argument('--model-dict', default='', type=str,
                    help='location to the configuration file that defines the model')
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--load-prune', default='', type=str, metavar='PATH',
                    help='Initialize model from unpruned checkpoint (default: none)')
parser.add_argument('--load-parent', default='', type=str, metavar='PATH',
                    help='Initialize model from parent checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--train-size', type=int, default=192, metavar='N',
                    help='train Image patch size (default: None => model default)')
parser.add_argument('--val-size', type=int, default=288, metavar='N',
                    help='validate Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='validation batch size override (default: None)')



# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=True,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')



parser.add_argument('--reset_bn', action='store_true', default=False,
                    help='batch norm calibrate')



# Misc
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=3, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)


    # net_dict= {'ks': [5, 7, 3, 5, 3, 5, 7, 3, 5, 3, 3, 7, 5, 7, 7, 5], 'e': [6, 6, 4, 6, 3, 3, 6, 4, 6, 6, 6, 4, 4, 4, 6, 3], 'd': [3, 3, 4, 4, 2], 'r': 192}

    # net_dict={'ks': [5, 7, 3, 5, 3, 5, 7, 5, 5, 3, 5, 7, 5, 7, 7], 'e': [6, 6, 4, 6, 3.0, 2.4375, 6, 3.0, 4.5, 6, 4.5, 4, 3, 3.25, 4.5], 'd': [3, 3, 4, 3, 2], 'r': 192}

    # net_dict={'ks': [5, 7, 3, 5, 3, 5, 7, 5, 5, 3, 5, 7, 5, 7, 7], 'e': [6, 4.875, 4, 6, 2.625, 3, 4.875, 4, 5.25, 6, 6, 4.0, 3, 4, 6], 'd': [3, 3, 4, 3, 2], 'r': 192}

    # net_dict={'ks': [5, 7, 3, 5, 3, 5, 7, 5, 5, 3, 5, 7, 5, 7, 7], 'e': [6, 5.25, 3.75, 6, 3, 2.25, 6, 3.0, 6, 4.5, 6, 3.75, 3, 4.0, 4.5], 'd': [3, 3, 4, 3, 2], 'r': 192}
    net_dict = json.load(open(args.model_dict))
    # evaluator = OFAEvaluator(model_path='OFA/ofa_mbv3_d234_e346_k357_w1.2')
    # model, arch_config = evaluator.sample(net_dict)
    # if args.local_rank == 0:
    #     print('to build:', subnet.config)
    # net_config = json.load(open(args.model_config))
    model = MobileNetV3(net_dict)
    # init = torch.load(args.initial_checkpoint, map_location='cpu')['state_dict']
    # model.load_state_dict(init)


    if args.load_parent:
        parent_init_path = './output/' + args.load_parent + '/model_best.pth.tar'
        # parent_init_path =  args.load_parent
        parent_init = torch.load(parent_init_path, map_location='cpu')['state_dict']
        new_state_dict = model.state_dict()
        for k, v in parent_init.items():
            # strip `module.` prefix
            name = k[7:] if k.startswith('module') else k
            if name in new_state_dict.keys() and v.size()== new_state_dict[name].size():
                if args.local_rank == 0:
                    print('yes,copied!')
                    print(name)
                    print(v.size())
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


    if args.load_prune:
        old_config_path = './output/' + args.load_prune + '/net.dict'
        old_init_path = './output/' + args.load_prune + '/model_best.pth.tar'
        old_model_config = json.load(open(old_config_path))
        old_model = MobileNetV3(old_model_config)
        old_init = torch.load(old_init_path, map_location='cpu')['state_dict']
        old_model.load_state_dict(old_init)
        model = prune_lowindex(old_model, model)
        del old_model





    if args.local_rank == 0:
        _logger.info(
            f'Model created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), verbose=args.local_rank == 0)

    # move model to GPU, enable channels last layout if set
    model.cuda()

    

    # setup automatic mixed-precision (AMP) loss scaling and op casting

    amp_autocast = torch.cuda.amp.autocast

   


    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        batch_size=args.batch_size)

    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        batch_size=args.batch_size)




    # create data loaders w/ augmentation pipeiine
    loader_train = create_loader(
        dataset_train,
        input_size=args.train_size,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,


    )


    loader_eval = create_loader(
        dataset_eval,
        input_size=args.val_size,
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
    )

    # setup loss function

    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking

    if args.reset_bn:
        print('calibrate now...')
        calibrate(model, loader_train)


    eval_top1 = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
    if args.local_rank == 0:
        print('*eval top1* :',eval_top1)

    # if not os.path.exists(args.experiment):
    #     os.makedirs(args.experiment,exist_ok=True)

    net_name=args.experiment.split('/')[-1]
    net_pre=args.experiment[:-2]
    job = os.path.join(net_pre, "{}.states".format(net_name))

    # pre_job=os.path.join(net_pre,'net_{}.dict'.format(net_name))
    # save = json.load(open(pre_job))
    # save['metric']=eval_top1

    with open(job, 'w') as handle:
        # json.dump(eval_metrics, handle)
        json.dump({'metric': eval_top1, 'arch': json.load(open(args.model_dict))}, handle)
        # json.dump(save, handle)







def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    # metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return top1_m.avg


if __name__ == '__main__':
    main()
