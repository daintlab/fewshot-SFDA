for lr in 1e-5
do
    for shot in 5 10
    do
        for oda_seed in 2020
        do
            for seed in 0 1 2
            do  
                CUDA_VISIBLE_DEVICES=$3 python target_finetune.py --dataset VISDA-C \
                --pretrain $2 --adapt $1 --few_shot $shot \
                --work_dir $2'_VISDA-C_'$1'_'$shot'shot_SAM_fixedval_lr_'$lr'_subset_'$oda_seed'_'$seed --lr $lr --source 0 --seed $seed --SAM --subset --oda_seed $oda_seed &
                wait
            done
        done
    done
done
# done