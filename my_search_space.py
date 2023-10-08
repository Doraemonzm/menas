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
        self.stride=[2, 2, 2, 2, 1]
        self.width_mult=[0.5, 0.625, 0.75, 1, 1.25, 1.5, 2]
        self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
        self.exp_ratio = [1, 2, 3, 4, 5, 6]  # expansion rate
        self.depth = [1, 2, 3, 4, 5, 6]  # number of Inverted Residual Bottleneck layers repetition
        self.use_se=[1, 0]


        self.flop_budget=[100, 1600]

        self.prune_rate=[0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975]





        self.mutate_ks_prob= 0.04
        self.mutate_depth_prob = 0.05
        self.mutate_exp_prob = 0.04
        self.mutate_blkmul_prob = 0.1
        self.mutate_se_prob = 0.03
        self.mutate_stride_prob = 0.35




    def sample(self, n_samples=1):
        """ randomly sample a architecture"""
        ns=self.num_stages
        s = self.stride
        ks = self.kernel_size
        e = self.exp_ratio
        d = self.depth
        # r = self.resolution
        wm=self.width_mult
        us=self.use_se


        data = []


        for n in range(n_samples):
            # first sample layers
            flag=False

            while not flag:

                stride=np.random.permutation(s).tolist()

                depth = np.random.choice(d, ns, replace=True).tolist()
                width_mul = np.random.choice(wm, ns, replace=True).tolist()


                # then sample kernel size, expansion rate and resolution
                kernel_size = np.random.choice(ks, size=int(np.sum(depth)), replace=True).tolist()
                # resolution = int(np.random.choice(r))
                exp_ratio = np.random.choice(e, size=int(np.sum(depth)), replace=True).tolist()



                se_use=np.random.choice(us, size=int(np.sum(depth)), replace=True).tolist()




                net_dict = {'ks': kernel_size, 'e': exp_ratio, 'd': depth, 's':stride, 'width-mul': width_mul, 'se':se_use}

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
        se=new_config['se']
        assert len(kernel_size)==len(exp_ratio)==len(se)
        complete = sum(depth) - len(kernel_size)
        if complete==0:
            # print('it is valid!')
            # print(new_config)
            return new_config

        elif complete>0:
            # print('Not valid! Depth larger than ks, exp, make it valid now!')
            complete_ks=np.random.choice(self.kernel_size, size=complete, replace=True).tolist()
            complete_exp = np.random.choice(self.exp_ratio, size=complete, replace=True).tolist()
            complete_se = np.random.choice(self.use_se, size=complete, replace=True).tolist()
            new_config['ks']+=complete_ks
            new_config['e'] += complete_exp
            new_config['se']+=complete_se
        else:
            del new_config['ks'][complete:]
            del new_config['e'][complete:]
            del new_config['se'][complete:]

        return new_config


    def check_prune(self, old_cfg, p_cfg):
        old_model=MobileNetV3(old_cfg)
        new_model=MobileNetV3(p_cfg)
        old_state_dict = old_model.state_dict()
        new_state_dict = new_model.state_dict()
        for k, v in old_state_dict.items():
            if k in new_state_dict.keys():
                if len(v.size()) != len(new_state_dict[k].size()):
                    return False
        return True


    def gen_prune(self, config, shrink_num):
        res=[]
        for i in range(shrink_num):
            flag = False
            while not flag:
                temp = copy.deepcopy(config)
                # print('before shrink:------------------------')
                # print(temp)
                exp_ratio = temp['e']
                for idx, e in enumerate(exp_ratio):
                    if random.random()>0.5:
                        shrink_choice=np.random.choice(self.prune_rate)
                        e=e*shrink_choice
                        exp_ratio[idx]=e
                temp['e']=exp_ratio
                if temp not in res and temp!=config and self.check_prune(config, temp):
                    res.append(temp)
                    flag=True
        return res




    def mutate(self, config):
        while True:
            new_config = copy.deepcopy(config)


            for i in range(len(config['ks'])):
                if random.random() < self.mutate_ks_prob:
                    print('mutate kernel size:')
                    new_config['ks'][i] = random.choice(self.kernel_size)

            for i in range(len(config['e'])):
                if random.random() < self.mutate_exp_prob:
                    print('mutate expand rate:')
                    new_config['e'][i] = random.choice(self.exp_ratio)

            for i in range(len(config['d'])):
                if random.random() < self.mutate_depth_prob:
                    print('mutate depth:')
                    new_config['d'][i] = random.choice(self.depth)

            if random.random()<self.mutate_stride_prob:
                print('mutate stride list:')
                new_config['s']=np.random.permutation(config['s']).tolist()

            for i in range(len(config['width-mul'])):
                if random.random() < self.mutate_blkmul_prob:
                    print('mutate width mult:')
                    new_config['width-mul'][i] = random.choice(self.width_mult)

            for i in range(len(config['se'])):
                if random.random() < self.mutate_se_prob:
                    print('mutate se:')
                    new_config['se'][i] = 1-config['se'][i]



            valid_new_config= self.make_config_valid(new_config)
            model=MobileNetV3(valid_new_config)
            new_flop, new_param= cal_complexity(model)
            print('new complexity: flop:{}  param:{}'.format(new_flop, new_param))
            if  valid_new_config!=config and self.flop_budget[0]<= new_flop <= self.flop_budget[1]:
                return valid_new_config






if __name__ == '__main__':

    search_space = SearchSpace()
    new_dict=search_space.sample(10)
    for item in new_dict:
        print(item)


