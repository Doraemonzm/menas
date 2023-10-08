import numpy as np
import torch
import torch.nn as nn
import subprocess
import os
import json
import glob


VAL_CFG = {
     'batch-size': 512,  'workers': 2, 'val-size': 224, 'load-prune': None, 'load-parent':None, 'reset_bn':True,
  'amp': True,  'experiment': None
}




def calibrate(model, train_loader):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)
            m.momentum = 0.1
    model.train()
    with torch.no_grad():
        for i, (images, target) in enumerate(train_loader):
            if i >= 200:
                break
            images = images.cuda()
            logits = model(images)
            del logits





def prune_lowindex(old_model, new_model):
    end=None
    for [m0, m1] in zip(old_model.modules(), new_model.modules()):
    # for [m0, m1] in zip(old_model.features, new_model.features):
        if isinstance(m0, nn.Conv2d) and isinstance(m1, nn.Conv2d):
            # print('copy conv!')
            start = m1.in_channels
            end = m1.out_channels
            # print('start:', start)
            # print('end:', end)
            w1 = m0.weight.data[:end, :start, :, :].clone()
            m1.weight.data = w1.clone()
            if m0.bias is None:
                m1.bias=None
            else:
                m1.bias.data = m0.bias.data[:end].clone()

        elif isinstance(m0, nn.BatchNorm2d) and isinstance(m1, nn.BatchNorm2d):
            # print('start batchnorm copy:')

            m1.weight.data = m0.weight.data[:end].clone()
            m1.bias.data = m0.bias.data[:end].clone()
            m1.running_mean = m0.running_mean[:end].clone()
            m1.running_var = m0.running_var[:end].clone()

        #
        elif isinstance(m0, nn.Linear) and isinstance(m1, nn.Linear):
            # print('copy linear!')
            start = m1.in_features
            end = m1.out_features


            m1.weight.data = m0.weight.data[:end, :start].clone()
            if m0.bias is None:
                m1.bias=None
            else:
                m1.bias.data = m0.bias.data[:end].clone()

    return new_model


def prepare_distribute_val(xargs, prune_candidate, save, load_prune = None, load_parent = None):


    bash_file = ['#!/bin/bash']
    for idx, cand in enumerate(prune_candidate):
        job = os.path.join(save, "net_{}.dict".format(idx))
        with open(job, 'w') as handle:
            json.dump(cand, handle)
        execution_line = "CUDA_VISIBLE_DEVICES={}  python -m torch.distributed.launch  --nproc_per_node=1  --master_port {}  individual_validate.py  {}".format(int(idx%8), 47769+int(idx%8), xargs.sub_val_set)
        VAL_CFG['model-dict']=job
        VAL_CFG['experiment']=save+'/'+ str(idx)
        VAL_CFG['load-prune'] = load_prune
        VAL_CFG['load-parent'] = load_parent
        # VAL_CFG['val-size'] = cand['r']
        for k, v in VAL_CFG.items():
            if v is not None:
                if isinstance(v, bool):
                    if v:
                        execution_line += " --{}".format(k)
                else:
                    execution_line += " --{} {}".format(k, v)
        execution_line += ' &'
        bash_file.append(execution_line)
        if (idx + 1) % 8 == 0:
            bash_file.append('wait')
        # bash_file.append('wait')

    with open(os.path.join(save, 'run_bash.sh'), 'w') as handle:
        for line in bash_file:
            handle.write(line + os.linesep)



def prune_filter(SS, individual, xargs):
    prune_candidate= SS.gen_prune(individual['arch'], xargs.shrink_num)
    save_path= os.path.join('./output/'+individual['save'],'prune_candidate')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    prepare_distribute_val(xargs, prune_candidate, save_path, load_prune=individual['save'])
    res=[]
    subprocess.call("sh {}/run_bash.sh".format(save_path), shell=True)

    for file in glob.glob(os.path.join(save_path, '*.states',)):  # glob.glob 返回所有匹配的文件路径列表
        cand = json.load(open(file))
        res.append(cand)
    # print('res:',res)

    sorted_res=sorted(res, key=lambda x: x['metric'],reverse=True)[:xargs.actual_prune]
    # print('sorted res:', sorted_res)
    return [p['arch'] for p in sorted_res]
