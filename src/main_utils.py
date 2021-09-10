import logging
import torch
import torch.nn as nn
import os
import time
import numpy as np
import random

def append_to_file(v,filename):
    with open('/home/antpc/Desktop/Seq2BashAST/' + filename,'a+') as f:
        print(v,file=f)

def check_path(path, exist_ok=False):
    """Check if `path` exists, makedirs if not else warning/IOError."""
    if os.path.exists(path):
        if not exist_ok:
            raise IOError(f"path {path} exists, stop.")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def get_logger(name=__file__, level=logging.INFO,filename='log/seq2bash_explain.log'):
    check_path(filename, exist_ok=True)
    class RunLogger(logging.LoggerAdapter):
        """
        This example adapter expects the passed in dict-like object to have a
        'runid' key, whose value in brackets is prepended to the log message.
        """
        def process(self, msg, kwargs):
            return '[%s] %s' % (self.extra['runid'], msg), kwargs

    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        runlogger = RunLogger(logger,{'runid' : logger.runid})
        return runlogger
        #return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.runid = str(time.time()).replace('.','')
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s||%(module)s||%(lineno)d||%(levelname)s||%(message)s",)
    handler = logging.FileHandler(filename,mode='a')
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)
    runlogger = RunLogger(logger,{'runid':logger.runid})
    return runlogger

logger = get_logger()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_device(args):
    if args.device not in ['gpu','cpu','auto']:
        logger.warning(f'Device option not supported. Received {args.device}')
        return args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(args.device == 'gpu' and not torch.cuda.is_available()):
        logger.error('Backend device: %s not available',args.device)
    if args.device != 'auto':
        device = torch.device('cpu'  if args.device=='cpu' else 'cuda')
    args.device=device
    logger.info(f'Using device {args.device}')
    return device


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m,ignore_list = []):
    ignore_list = ignore_list + ['layer_norm']
    for name, param in m.named_parameters():
        continue_flag = False
        for listitem in ignore_list:
            if listitem in name:
                logger.info(f'Ignoring initialisation for {name}')
                continue_flag = True
                break
        if continue_flag:
            continue
        if 'weight' in name:
            logger.info(f'Initialising {name} withn Kaiming Normal Distribution, Shape: {param.data.shape}')
            # nn.init.normal_(param.data, mean=0, std=0.01)
            nn.init.kaiming_normal_(param.data)
        else:
            logger.info(f'Initialising {name} with Zero')
            nn.init.constant_(param.data, 0)


def memReport():
    import gc
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj),obj.size())


def cpuStats():
    import os
    import sys
    import psutil
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    print(memoryUse)


def str2bool(v):
    return v.lower() in ('true')


def str2tuple(v):
    v = v.split('`!`!`')
    return (v[0],v[1])


def set_random_seed(seed, is_cuda):
    """Sets the random seed."""
    if seed > 0:
        torch.manual_seed(seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)

    if is_cuda and seed > 0:
        # These ensure same initialization in multi gpu mode
        torch.cuda.manual_seed(seed)

