## Getting Started

Download the CIFAR-10 and CIFAR-100 datasets and place them in the data folder.

The saved ImageNet pre-trained model is palced in` ./ `  

### CIFAR-10

```Python
python train_cifar10.py    --batch-size  96  --checkpoint /pretrained_model_path/    --autoaug  autoaug_cifar10  --cutmix  --cutmix_prob 0.5   --random_erase  --lr  0.0005  --soft_label  --epochs 100  
```

### CIFAR-100

```Python
python train_cifar100.py    --batch-size  96    --checkpoint /pretrained_model_path/    --autoaug  autoaug_cifar10   --cutmix   --cutmix_prob 0.5  --random_erase   --soft_label  --lr  0.0005 --epochs 100   --clip
```

