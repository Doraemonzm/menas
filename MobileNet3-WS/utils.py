import os
import copy
import json
import yaml
import numpy as np

from torchprofile import profile_macs

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn






def count_parameters(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params



def cal_flops(config):
    from torchprofile import profile_macs
    from My_Mobilenet3 import MobileNetV3
    model = MobileNetV3(config)
    inputs = torch.randn(1, 3, 224, 224)
    net_flop = int(profile_macs(copy.deepcopy(model), inputs))
    return (net_flop/ 1e6)




def cal_params(config):

    from My_Mobilenet3 import MobileNetV3
    model = MobileNetV3(config)
    net_param=count_parameters(model)
    return (net_param / 1e6)