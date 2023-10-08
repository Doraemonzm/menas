
import random
import os
import subprocess
import sys


import time,argparse
import torch
from my_search_space import SearchSpace
from utils import cal_flops,  cal_params


from prune_helper import prune_filter

import numpy as np
import json

import copy

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

CFG = {
    'model-config': None, 'init': None, 'batch-size': 128,  'train-size': 224, 'val-size': 224, 'workers': 8,  'epochs': 30, 'load-prune': None, 'load-parent':None,
    'sched': 'step', 'decay-epochs': 2.4, 'decay-rate': 0.97,  'opt': 'rmsproptf', 'opt-eps': 0.001,
    'warmup-lr': 1e-6, 'weight-decay': 1e-5, 'drop-path': 0.2, 'amp': True,  'lr': 0.064, 'experiment': None
}





def exec_distribute_train(xargs, config_path, epoch, indiv_dict,  save, load_prune = None, load_parent = None):
    bash_file = ['#!/bin/bash']
    execution_line = "sh scripts/distributed_train.sh  {}  {}".format(xargs.gpu_num, xargs.data_dir)
    CFG['model-config']=config_path
    CFG['experiment']=save
    CFG['epochs'] = epoch
    CFG['load-prune'] = load_prune
    CFG['load-parent'] = load_parent
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
        os.makedirs(save,exist_ok=True)
    with open(os.path.join(save, 'run_bash.sh'), 'w') as handle:
        for line in bash_file:
            handle.write(line + os.linesep)
    subprocess.call("sh {}/run_bash.sh".format(save), shell=True)


def obtain_indiv_acc(save_path):
    import pandas as pd
    data = pd.read_csv(save_path)['eval_top1']
    final_list = list(data)
    return max(final_list)





def next_generation(SS, parent, xargs, copy_popl):
    already=[al['arch'] for  al in copy_popl]
    res=[]
    for config in parent:
        for i in range(xargs.mutate_num):
            flag = False
            while not flag:
                mu=SS.mutate(config['arch'])
                generate=[g['arch'] for g in res]
                if mu not in already and mu not in generate:
                    b={}.fromkeys(('arch', 'save', 'acc', 'time', 'epoch'))
                    b['arch']=mu
                    b['parent']=config['save']
                    b['epoch']=xargs.finetune_epoch
                    res.append(b)
                    flag=True
    return res




def train_one_indiv(xargs, individual, idx, gen_id, prune_idx=None, load_prune=None, load_parent=None):
    str_index = 'gen-{}_{}'.format(gen_id, idx)
    indiv_dict = individual['arch']
    print('indiv dict is:', indiv_dict)

    if prune_idx==None:
        save = './output/'+ str_index
        print('saving is:', save)
    else:
        save=  './output/'+ str_index + '/pruned_{}'.format(prune_idx)
        print('saving for pruned is:', save)
    if not os.path.exists(save):
        os.makedirs(save, exist_ok=True)
    # subnet, arch_config = evaluator.sample(indiv_dict)
    job = os.path.join(save, "net.dict")
    print('job is:', job)

    with open(job, 'w') as handle:
        json.dump(indiv_dict, handle)

    t1 = time.time()
    exec_distribute_train(xargs, job, individual['epoch'], indiv_dict, save=save, load_prune=load_prune, load_parent= load_parent)
    t2 = time.time()
    t = t2 - t1
    individual['save'] = str_index
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




def initial_pop(SS , xargs, specific=None):
    print('***********************************************')
    print('Initial evolution :')
    pruned_popl, popl = [], []
    if xargs.resume_path:
        print('resuming from {}-th individual:'.format(specific['indiv_id']))
        popl = specific['popl']
        pruned_popl= specific['pruned']


    else:
        temp_pop = SS.sample(xargs.n_doe)


        for idx, arch_dict in enumerate(temp_pop):
            a = {}.fromkeys(('arch', 'save', 'acc', 'time', 'epoch'))
            a['arch'] = arch_dict
            a['epoch']=xargs.initial_epoch
            popl.append(a)



    print('before train, pop:')
    for item in popl:
        print(item)


    for idx, individual in enumerate(popl):

        if specific and idx < specific['indiv_id']:
            continue
        print('Train begin:\n')

        individual=train_one_indiv(xargs, individual, idx, gen_id=1)

        print('Generation {}: The {}-th individual train Finished:'.format(1, idx))
        print(individual['arch'])
        print('Accuracy: {}, Time Cost: {}, FLOPs: {} M\n'.format(individual['acc'], individual['time'], individual['Flops']))

        print('Generate prune candidates for {}-th indiv:'.format(idx))
        individual_pruned = prune_filter(SS, individual, xargs)
        print('Finish\n')

        for p_idx, ind_p in enumerate(individual_pruned):
            p = {}.fromkeys(('arch', 'save', 'acc', 'time', 'epoch'))
            p['arch'] = ind_p
            p['epoch'] = xargs.prune_epoch

            p_individual = train_one_indiv(xargs, p, idx, gen_id=1, prune_idx = p_idx, load_prune=individual['save'])
            pruned_popl.append(p_individual)
            print('Generation {}: The {}-th pruned of {}-th individual:'.format(1, p_idx, idx))
            print(p_individual['arch'])
            print('Accuracy: {}, Time Cost: {}, FLOPs: {} M\n'.format(p_individual['acc'], p_individual['time'],
                                                                      p_individual['Flops']))

        if (idx+1) % xargs.save_freq==0 :
            print('saving regularly..')
            resume_state = {
                'popl': popl,
                'pruned':pruned_popl,
                'gen_id':1,
                'indiv_id':idx+1
            }
            torch.save(resume_state, os.path.join('./output', 'gen_{}_resume.pth'.format(1)))


    print('Generation 1 finished!')
    total_score = [p['acc'] for p in popl] + [pp['acc'] for pp in pruned_popl]
    total_arch= [p['arch'] for p in popl] + [pp['arch'] for pp in pruned_popl]
    print('total score is:', total_score)
    print('tital arch is:', total_arch)
    best_index = total_score.index(max(total_score))
    total_time = [t['time'] for t in popl] + [tt['time'] for tt in pruned_popl]
    print('Best acc: {}'.format(max(total_score)))
    print('Best param: {}'.format(total_arch[best_index]))
    print('Total search time for Gen 1 is: {} hour'.format(sum(total_time) / 3600))


    print('Saving for generation 1:')
    with open(os.path.join('./output', "gen_{}.stats".format(1)), "w") as handle:
        json.dump({'popl': popl, 'pruned': pruned_popl}, handle)
    print('***********************************************')



def evo_gen(SS ,xargs, start_gen_id, specific=None):
    for gen_id in range(start_gen_id, xargs.gene_num+1):
        print('*********************************************************')
        print('Generation {} :'.format(gen_id))

        print('loading from last generation:')
        save= json.load(open('./output/gen_{}.stats'.format(gen_id-1)))
        popl = save['popl']
        pruned_popl= save['pruned']

        archive=popl+pruned_popl


        if specific and gen_id==specific['gen_id']:
            print('resuming from {}-th individual:'.format(specific['indiv_id']))
            archieve_popl=specific['this_popl']
            archieve_pruned=specific['this_pruned']


        else:
            archieve_popl=[]
            archieve_pruned=[]

            # prepare1 = sorted(popl, key=lambda x: x['acc'], reverse=True)[:xargs.unpruned_num]
            prepare2 = sorted(pruned_popl, key=lambda x: x['acc'], reverse=True)[:xargs.pruned_num]

            prepare3=sorted(popl+pruned_popl, key=lambda x: x['Flops'], reverse=True)[:xargs.flop_num]


            parent= prepare2 + prepare3

            print('selected candidates by accuracy:')
            for item in prepare2:
                print(item['arch'], item['acc'])
            print('selected candidates by flops:')
            for item in prepare3:
                print(item['arch'], item['acc'], item['Flops'])

            copy_archieve = copy.deepcopy(archive)

            derive_popl = next_generation(SS, parent, xargs, copy_archieve)
            print('new generated candidates in gen {}:'.format(gen_id))
            for d_item in derive_popl:
                print(d_item['arch'])




            archieve_popl.extend(derive_popl)

            print('*********************************************************')



        print('before train, pop:',len(archieve_popl))
        # assert len(derive_pop)==xargs.popl_size
        for item in archieve_popl:
            print(item['arch'])


        for idx, cand in enumerate(archieve_popl):
            if specific and gen_id==specific['gen_id'] and idx< specific['indiv_id']:
                    continue

            search_t1=time.time()
            individual = train_one_indiv(xargs, cand, idx, gen_id=gen_id, load_parent=cand['parent'])

            print('Generation {}: The {}-th individual train Finished:'.format(gen_id, idx))
            print(individual['arch'])
            print('Accuracy: {}, Time Cost: {}, FLOPs: {} M\n'.format(individual['acc'], individual['time'], individual['Flops']))

            print('Generate prune candidates for {}-th indiv:'.format(idx))
            individual_pruned = prune_filter(SS, individual, xargs)
            print('Finish\n')

            for p_idx, ind_p in enumerate(individual_pruned):
                p = {}.fromkeys(('arch', 'save', 'acc', 'time', 'epoch'))
                p['arch'] = ind_p
                p['epoch'] = xargs.prune_epoch
                p_individual = train_one_indiv(xargs, p, idx, gen_id=gen_id, prune_idx=p_idx, load_prune=cand['save'])
                archieve_pruned.append(p_individual)
                print('Generation {}: The {}-th pruned of {}-th individual:'.format(gen_id, p_idx, idx))
                print(p_individual['arch'])
                print('Accuracy: {}, Time Cost: {}, FLOPs: {} M\n'.format(p_individual['acc'], p_individual['time'],
                                                                          p_individual['Flops']))


            search_t2=time.time()
            print('search for one individual and its three pruned costs: ', search_t2-search_t1)


            print('***********************************************')
            if (idx+1) % xargs.save_freq == 0:
                print('saving regularly..')
                resume_state = {
                    'popl': popl+archieve_popl,
                    'pruned': pruned_popl+ archieve_pruned,
                    'this_popl': archieve_popl,
                    'this_pruned': archieve_pruned,
                    'gen_id': gen_id,
                    'indiv_id':idx+1,

                }
                torch.save(resume_state, os.path.join('./output', 'gen_{}_resume.pth'.format(gen_id)))

        print('Generation {} finished!'.format(gen_id))
        o_score = [up['acc'] for up in archieve_popl] + [upp['acc'] for upp in archieve_pruned]
        o_arch = [a['arch'] for a in archieve_popl] + [aa['arch'] for aa in archieve_pruned]
        o_index=o_score.index(max(o_score))

        o_time= [ot['time'] for ot in archieve_popl] + [ott['time'] for ott in archieve_pruned]

        print('Best acc: {}'.format(max(o_score)))
        print('Best param: {}'.format(o_arch[o_index]))
        print('Total search time for Gen {} is: {} hour'.format(gen_id, sum(o_time)/3600))


        print('Saving for generation {}:'.format(gen_id))
        with open(os.path.join('./output', "gen_{}.stats".format(gen_id)), "w") as handle:
            json.dump({'popl': popl + archieve_popl, 'pruned': pruned_popl + archieve_pruned,
                       'candidates': archieve_popl + archieve_pruned,
                       }, handle)
        print('***********************************************')


def resume(xargs, SS):

    resume_save=torch.load(xargs.resume_path)
    gen_id=resume_save['gen_id']

    if gen_id==1:

        initial_pop(SS,xargs,resume_save)
        evo_gen(SS,xargs,start_gen_id=2)

    elif gen_id>=2:
        evo_gen(SS, xargs, start_gen_id=gen_id, specific=resume_save)




def main(xargs):

    SS= SearchSpace()

    if xargs.resume_path:
        resume(xargs, SS)
    else:
        if xargs.start_gen == None or xargs.start_gen == 1:
            initial_pop(SS , xargs)
            evo_gen(SS, xargs, start_gen_id=2)
        elif xargs.start_gen>=2:
            evo_gen(SS, xargs, start_gen_id=xargs.start_gen)

    final=json.load(open('./output/gen_{}.stats'.format(xargs.gene_num)))
    all = final['popl'] + final['pruned']
    final_score = [f['acc'] for f in all]
    final_time = [t['time'] for t in all]
    best_index = final_score.index(max(final_score))
    best_param = all[best_index]['arch']
    print('Finished!')
    print('Final best acc is: ', max(final_score))
    print('Best param is:', best_param)
    print('Total search time is: {} hours'.format(sum(final_time) / 3600))



#
if __name__ == "__main__":
    parser = argparse.ArgumentParser("evolution search without online-predictor")

    # evolution
    parser.add_argument('--n_doe', default=4, type=int, help='individual number')
    # parser.add_argument('--reduced_doe', default=30, type=int, help='select for parent')
    parser.add_argument('--mutate_num', default=1, type=int, help='number of mutate')
    parser.add_argument('--gene_num', default=5, type=int, help='generation number')
    parser.add_argument('--pruned_num', default=3, type=int, help='generation number')
    # parser.add_argument('--unpruned_num', default=10, type=int, help='generation number')
    parser.add_argument('--flop_num', default=1, type=int, help='generation number')

    # resume
    parser.add_argument('--start_gen', default=1, type=int, help='generation to resume')
    parser.add_argument('--save_freq', default=1, type=int, help='save intervel')
    parser.add_argument('--resume_path', default='', type=str, help='resume model path')


    # prune
    parser.add_argument('--shrink_num', default=8, type=int, help='prune candidate number')
    parser.add_argument('--actual_prune', default=1, type=int, help='actual prune number')

    # individual
    parser.add_argument('--gpu_num', default=8, type=int, help='gpu number')
    parser.add_argument('--initial_epoch', default=1, type=int, help='epoch number of generation 1')
    parser.add_argument('--finetune_epoch', default=1, type=int, help='epoch number of generation 1')
    parser.add_argument('--prune_epoch', default=1, type=int, help='epoch number of generation 1')
    parser.add_argument('--data_dir', type=str, default="",
                        help='location of the data corpus')
    parser.add_argument('--sub_val_set',type=str, default="", help='location of the sub validation data, used to select Lottery Tickets')
    # parser.add_argument('--rand_seed', type=int, help='manual seed')
    args = parser.parse_args()
    for k, v in sorted(vars(args).items()):
        print(k,' = ',v)
    main(args)



