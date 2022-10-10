#!/bin/bash
# train certain dataset in single-domain setting
# $1 : dataset name = ['PACS','office_home','domain_net']
# $2 ~ : gpu idx

if [ $1 = 'domain_net' ]
then
    CUDA_VISIBLE_DEVICES=$2 python ../train.py --dataset $1 --target 0 \
    --source 0 --lr 5e-05 --wd 1e-06 --ratio 0.2 --seed 0 --steps 15000 --val_freq 1000 \
    --save_path '../ckpts/'$1'_single_source/source_0' &
    
    CUDA_VISIBLE_DEVICES=$3 python ../train.py --dataset $1 --target 0 \
    --source 1 --lr 5e-05 --wd 1e-06 --ratio 0.2 --seed 0 --steps 15000 --val_freq 1000 \
    --save_path '../ckpts/'$1'_single_source/source_1' &

    CUDA_VISIBLE_DEVICES=$4 python ../train.py --dataset $1 --target 0 \
    --source 2 --lr 5e-05 --wd 1e-06 --ratio 0.2 --seed 0 --steps 15000 --val_freq 1000 \
    --save_path '../ckpts/'$1'_single_source/source_2' &

    CUDA_VISIBLE_DEVICES=$5 python ../train.py --dataset $1 --target 0 \
    --source 3 --lr 5e-05 --wd 1e-06 --ratio 0.2 --seed 0 --steps 15000 --val_freq 1000 \
    --save_path '../ckpts/'$1'_single_source/source_3' &

    CUDA_VISIBLE_DEVICES=$6 python ../train.py --dataset $1 --target 0 \
    --source 4 --lr 5e-05 --wd 1e-06 --ratio 0.2 --seed 0 --steps 15000 --val_freq 1000 \
    --save_path '../ckpts/'$1'_single_source/source_4' &

    CUDA_VISIBLE_DEVICES=$7 python ../train.py --dataset $1 --target 0 \
    --source 5 --lr 5e-05 --wd 1e-06 --ratio 0.2 --seed 0 --steps 15000 --val_freq 1000 \
    --save_path '../ckpts/'$1'_single_source/source_5' &

    wait
else
    CUDA_VISIBLE_DEVICES=$2 python ../train.py --dataset $1 --target 0 \
    --source 0 --lr 5e-05 --wd 1e-06 --ratio 0.2 --seed 0 --steps 5000 --val_freq 200 \
    --save_path '../ckpts/'$1'_single_source/source_0' &

    CUDA_VISIBLE_DEVICES=$3 python ../train.py --dataset $1 --target 0 \
    --source 1 --lr 5e-05 --wd 1e-06 --ratio 0.2 --seed 0 --steps 5000 --val_freq 200 \
    --save_path '../ckpts/'$1'_single_source/source_1' &

    CUDA_VISIBLE_DEVICES=$4 python ../train.py --dataset $1 --target 0 \
    --source 2 --lr 5e-05 --wd 1e-06 --ratio 0.2 --seed 0 --steps 5000 --val_freq 200 \
    --save_path '../ckpts/'$1'_single_source/source_2' &

    CUDA_VISIBLE_DEVICES=$5 python ../train.py --dataset $1 --target 0 \
    --source 3 --lr 5e-05 --wd 1e-06 --ratio 0.2 --seed 0 --steps 5000 --val_freq 200 \
    --save_path '../ckpts/'$1'_single_source/source_3' &

    wait
    
fi