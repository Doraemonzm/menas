import numpy as np
import random
import copy
import torch


from My_Mobilenet3 import MobileNetV3

from torchprofile import profile_macs
from utils import count_parameters

def cal_complexity(model):
    input = torch.randn(1, 3, 224, 224)
    # output=model(input)
    net_flop = int(profile_macs(copy.deepcopy(model), input))
    net_param = count_parameters(model)
    return net_flop / 1e6, net_param/1e6

class SearchSpace:

    def __init__(self):
        self.num_stages = 5  # number of blocks
        self.stride=[2, 2, 2, 1, 2]
        self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
        self.exp_ratio = [3, 4,  6]  # expansion rate
        self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition


        self.flop_budget=[1000, 1600]





    def sample(self, n_samples=1):
        """ randomly sample a architecture"""
        ns=self.num_stages
        s = self.stride
        ks = self.kernel_size
        e = self.exp_ratio
        d = self.depth



        data = []


        for n in range(n_samples):
            # first sample layers
            flag=False

            while not flag:

                depth = np.random.choice(d, ns, replace=True).tolist()

                # then sample kernel size, expansion rate and resolution
                kernel_size = np.random.choice(ks, size=int(np.sum(depth)), replace=True).tolist()
                # resolution = int(np.random.choice(r))
                exp_ratio = np.random.choice(e, size=int(np.sum(depth)), replace=True).tolist()


                net_dict = {'ks': kernel_size, 'e': exp_ratio, 'd': depth}

                model = MobileNetV3(net_dict)
                model_flop, _= cal_complexity(model)
                print('model flop is:', model_flop)
                if self.flop_budget[0]<= model_flop <= self.flop_budget[1] and net_dict not in data:
                    flag=True
                    data.append(net_dict)
                else:
                    print('budget not satisfiled!')

        return data






    def make_config_valid(self, config):

        new_config= copy.deepcopy(config)
        depth = new_config['d']
        kernel_size = new_config['ks']
        exp_ratio = new_config['e']

        complete = sum(depth) - len(kernel_size)
        if complete==0:
            # print('it is valid!')
            # print(new_config)
            return new_config

        elif complete>0:
            # print('Not valid! Depth larger than ks, exp, make it valid now!')
            complete_ks=np.random.choice(self.kernel_size, size=complete, replace=True).tolist()
            complete_exp = np.random.choice(self.exp_ratio, size=complete, replace=True).tolist()
            new_config['ks']+=complete_ks
            new_config['e'] += complete_exp
        else:
            del new_config['ks'][complete:]
            del new_config['e'][complete:]

        return new_config






if __name__ == '__main__':

    search_space = SearchSpace()
    new_dict=search_space.sample(10)
    for item in new_dict:
        print(item)


