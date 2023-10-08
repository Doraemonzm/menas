### Extract subset of ImageNet

```Python
python subImageNet.py  --data-path  /your_data_path/
```

### Searching with 8 GPUs

`n_doe` is the initial population size in the evolutionary search, while `gene_num` is the generation number.  `pruned_num` and `flop_num` are numbers of best-performing Lottery Tickets and the largest computation models when formulating parent populations, respectively. `shrink_num` is the number of generated pruning proposals for each trained network.` actual_prune` is the number of Lottery Tickets selected from pruning proposals.  `data_dir` is the path of 10% ImageNet training set, while  `sub_val_set` is the path of  10% ImageNet val set.

```Python
python search.py  --n_doe 60  --gene_num 5  --pruned_num 24 --flop_num 6  --shrink_num 40  --actual_prune 2  --data_dir /your_10%_train_path/   --sub_val_set  /your_10%_val_path/ 
```

### Re-training with 8 GPUs

we present the searched model dictionary in `searched/net.dict`

```Python
sh scripts/retrain.sh 8 /your_imagenet_path/   --model_cfg ./searched/net.dict   --model /name_for_save/  -b 128 --sched step --epochs 300 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999  --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064  
```