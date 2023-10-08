from scipy.stats import stats
import random
import os
import subprocess
import sys


import time,argparse
import torch
from my_search_space import SearchSpace
from utils import cal_flops,  cal_params


import numpy as np
import json
import torch.backends.cudnn as cudnn
import copy

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

CFG = {
    'model-config': None,  'batch-size': 128,  'train-size': 224, 'val-size': 224, 'workers': 8,  'epochs': 30,
    'sched': 'step', 'decay-epochs': 2.4, 'decay-rate': 0.97,  'opt': 'rmsproptf', 'opt-eps': 0.001,
    'warmup-lr': 1e-6, 'weight-decay': 1e-5, 'drop-path': 0.2, 'amp': True,  'lr': 0.064, 'experiment': None
}





def exec_distribute_train(xargs, config_path, indiv_dict, save):
    bash_file = ['#!/bin/bash']
    execution_line = "sh scripts/distributed_train.sh  {}  {}".format(xargs.gpu_num, xargs.data_dir)
    CFG['model-config']=config_path
    CFG['experiment']=save
    CFG['epochs'] = xargs.num_epoch

    # CFG['train-size']=indiv_dict['r']
    # CFG['val-size'] = indiv_dict['r']
    for k, v in CFG.items():
        if v is not None:
            if isinstance(v, bool):
                if v:
                    execution_line += " --{}".format(k)
            else:
                execution_line += " --{} {}".format(k, v)
    # execution_line += ' &'
    bash_file.append(execution_line)
    # bash_file.append('wait')
    # path='./output/'+save
    if not os.path.exists(save):
        os.makedirs(save, exist_ok=True)
    with open(os.path.join(save, 'run_bash.sh'), 'w') as handle:
        for line in bash_file:
            handle.write(line + os.linesep)
    subprocess.call("sh {}/run_bash.sh".format(save), shell=True)


def obtain_indiv_acc(save_path):
    import pandas as pd
    data = pd.read_csv(save_path)['eval_top1']
    final_list = list(data)
    return max(final_list)




def train_one_indiv(xargs, individual, idx):

    indiv_dict = individual['arch']
    print('indiv dict is:', indiv_dict)
    save = xargs.save_dir + '/{}'.format(idx)
    print('save is:', save)
    if not os.path.exists(save):
        os.makedirs(save, exist_ok=True)
    job = os.path.join(save, "{}.dict".format(idx))
    print('job is:', job)

    with open(job, 'w') as handle:
        json.dump(indiv_dict, handle)

    t1 = time.time()
    exec_distribute_train(xargs, job,  indiv_dict, save=save)
    t2 = time.time()
    t = t2 - t1

    individual['acc'] = obtain_indiv_acc(save + '/summary.csv')
    individual['time'] = t
    flop_cost = cal_flops(individual['arch'])
    param_cost= cal_params(individual['arch'])
    individual['Flops'] = flop_cost
    individual['Params'] = param_cost
    job1 = os.path.join(save, "net.info")

    with open(job1, 'w') as handle:
        json.dump(individual, handle)

    return individual




def random_search(SS ,xargs, specific=None):

        if specific:
            print('resuming from {}-th individual:'.format(specific['indiv_id']))
            archieve_popl=specific['popl']
        else:
            archieve_popl=[]
            random_cands= SS.sample(n_samples=xargs.random_num)
            for item in random_cands:
                a = {}.fromkeys(('arch', 'save', 'acc', 'time', 'epoch'))
                a['arch'] = item
                a['epoch'] = xargs.num_epoch
                if a not in archieve_popl:
                    archieve_popl.append(a)

        print('before train, pop:',len(archieve_popl))
        for item in archieve_popl:
            print(item['arch'])


        for idx, cand in enumerate(archieve_popl):
            if specific and idx< specific['indiv_id']:
                    continue

            search_t1=time.time()
            individual = train_one_indiv(xargs, cand, idx)

            print('The {}-th individual train Finished:'.format(idx))
            print(individual['arch'])
            print('Accuracy: {}, Time Cost: {}, FLOPs: {} M, Params: {} M\n'.format(individual['acc'], individual['time'], individual['Flops'], individual['Params']))

            search_t2=time.time()
            print('search for one individual: ', search_t2-search_t1)

            print('***********************************************')
            if (idx+1) % xargs.save_freq == 0:
                print('saving regularly..')
                resume_state = {
                    'popl': archieve_popl,
                    'indiv_id':idx+1,
                }
                torch.save(resume_state, os.path.join(xargs.save_dir, 'random_resume.pth'))


        o_score = [up['acc'] for up in archieve_popl]
        o_arch = [a['arch'] for a in archieve_popl]
        o_index=o_score.index(max(o_score))

        o_time= [ot['time'] for ot in archieve_popl]

        print('Best acc: {}'.format(max(o_score)))
        print('Best param: {}'.format(o_arch[o_index]))
        print('Total search time for random search is: {} hour'.format(sum(o_time)/3600))
        print('***********************************************')




def main(xargs):

    SS= SearchSpace()

    if xargs.resume_path:
        resume_save=torch.load(xargs.resume_path)
        random_search(SS, xargs, specific=resume_save)
    else:
        random_search(SS ,xargs)





#
if __name__ == "__main__":
    parser = argparse.ArgumentParser("evolution search with online-predictor")

    # evolution
    parser.add_argument('--random_num', default=200, type=int, help='number of mutate')

    # resume
    parser.add_argument('--save_freq', default=1, type=int, help='save intervel')
    parser.add_argument('--resume_path', default='', type=str, help='resume model path')

    # train
    parser.add_argument('--gpu_num', default=8, type=int, help='gpu number')
    parser.add_argument('--num_epoch', default=30, type=int, help='epoch number of generation 1')
    parser.add_argument('--data_dir', type=str, default="/datasets/subImageNet/",
                        help='location of the data corpus')

    parser.add_argument('--save_dir', default='./output/ws_random_search', type=str, help='save output path')

    args = parser.parse_args()
    for k, v in sorted(vars(args).items()):
        print(k,' = ',v)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    main(args)



