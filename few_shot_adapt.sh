for shot in 1 3 5 10
do
    for seed in 0 1 2
    do
        CUDA_VISIBLE_DEVICES=4 python target_finetune.py --dataset $1 \
        --pretrain SHOT --adapt $2 --few_shot $shot \
        --work_dir 'SHOT_'$1'_'$2'_'$shot'shot_SAM_val_'$seed --lr $3 --source 0 --SAM --seed $seed &

        CUDA_VISIBLE_DEVICES=5 python target_finetune.py --dataset $1 \
        --pretrain SHOT --adapt $2 --few_shot $shot \
        --work_dir 'SHOT_'$1'_'$2'_'$shot'shot_SAM_val_'$seed --lr $3 --source 1 --SAM --seed $seed &

        CUDA_VISIBLE_DEVICES=6 python target_finetune.py --dataset $1 \
        --pretrain SHOT --adapt $2 --few_shot $shot \
        --work_dir 'SHOT_'$1'_'$2'_'$shot'shot_SAM_val_'$seed --lr $3 --source 2 --SAM --seed $seed &

        CUDA_VISIBLE_DEVICES=7 python target_finetune.py --dataset $1 \
        --pretrain SHOT --adapt $2 --few_shot $shot \
        --work_dir 'SHOT_'$1'_'$2'_'$shot'shot_SAM_val_'$seed --lr $3 --source 3 --SAM --seed $seed &

        wait
    done
done
