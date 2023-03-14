for lr in 1e-5
do
    for shot in 10
    do
        for seed in 0 1 2
        do
            CUDA_VISIBLE_DEVICES=$3 python target_finetune.py --dataset VISDA-C \
            --pretrain $2 --adapt $1 --few_shot $shot \
            --work_dir $2'_VISDA-C_'$1'_'$shot'shot_SAM_fixedval_imbalance_lr_'$lr'_'$seed --lr $lr --source 0 --seed $seed --SAM --imbalance &

            wait
        done
    done
done