for lr in 1e-5
do
    for shot in 10
    do
        for seed in 0 1 2
        do
            # CUDA_VISIBLE_DEVICES=$4 python target_finetune.py --dataset $1 \
            # --pretrain $2 --adapt $3 --few_shot $shot \
            # --work_dir $2'_'$1'_'$3'_'$shot'shot_SAM_fixedval_imbalance_lr_'$lr'_'$seed --lr $lr --source 0 --SAM --seed $seed --imbalance &

            CUDA_VISIBLE_DEVICES=$4 python target_finetune.py --dataset $1 \
            --pretrain $2 --adapt $3 --few_shot $shot \
            --work_dir $2'_'$1'_'$3'_'$shot'shot_SAM_fixedval_imbalance_lr_'$lr'_'$seed --lr $lr --source 1 --SAM --seed $seed --imbalance &

            CUDA_VISIBLE_DEVICES=$5 python target_finetune.py --dataset $1 \
            --pretrain $2 --adapt $3 --few_shot $shot \
            --work_dir $2'_'$1'_'$3'_'$shot'shot_SAM_fixedval_imbalance_lr_'$lr'_'$seed --lr $lr --source 2 --SAM --seed $seed --imbalance &

            CUDA_VISIBLE_DEVICES=$6 python target_finetune.py --dataset $1 \
            --pretrain $2 --adapt $3 --few_shot $shot \
            --work_dir $2'_'$1'_'$3'_'$shot'shot_SAM_fixedval_imbalance_lr_'$lr'_'$seed --lr $lr --source 3 --SAM --seed $seed --imbalance &

            wait
        done
    done
done