for lr in 1e-5
do
    for shot in 1 3 5
    do
        for seed in 0 1 2
        do
            if [ $1 = 'VISDA-C' ]
            then

                CUDA_VISIBLE_DEVICES=$3 python target_finetune.py --dataset $1 \
                --pretrain SHOT_LPFT --adapt $2 --few_shot $shot \
                --work_dir 'SHOT_LPFT_'$1'_'$2'_'$shot'shot_SAM_fixedval_lr_'$lr'_'$seed --lr $lr --source 0 --SAM --seed $seed &
                wait
            elif [ $1 = 'office31' ]
            then
                CUDA_VISIBLE_DEVICES=$3 python target_finetune.py --dataset $1 \
                --pretrain SHOT_LPFT --adapt $2 --few_shot $shot \
                --work_dir 'SHOT_LPFT_'$1'_'$2'_'$shot'shot_SAM_fixedval_lr_'$lr'_'$seed --lr $lr --source 0 --SAM --seed $seed &

                CUDA_VISIBLE_DEVICES=$4 python target_finetune.py --dataset $1 \
                --pretrain SHOT_LPFT --adapt $2 --few_shot $shot \
                --work_dir 'SHOT_LPFT_'$1'_'$2'_'$shot'shot_SAM_fixedval_lr_'$lr'_'$seed --lr $lr --source 1 --SAM --seed $seed &

                CUDA_VISIBLE_DEVICES=$5 python target_finetune.py --dataset $1 \
                --pretrain SHOT_LPFT --adapt $2 --few_shot $shot \
                --work_dir 'SHOT_LPFT_'$1'_'$2'_'$shot'shot_SAM_fixedval_lr_'$lr'_'$seed --lr $lr --source 2 --SAM --seed $seed &
                wait
            else
                CUDA_VISIBLE_DEVICES=$3 python target_finetune.py --dataset $1 \
                --pretrain SHOT_LPFT --adapt $2 --few_shot $shot \
                --work_dir 'SHOT_LPFT_'$1'_'$2'_'$shot'shot_SAM_fixedval_lr_'$lr'_'$seed --lr $lr --source 0 --SAM --seed $seed &

                CUDA_VISIBLE_DEVICES=$4 python target_finetune.py --dataset $1 \
                --pretrain SHOT_LPFT --adapt $2 --few_shot $shot \
                --work_dir 'SHOT_LPFT_'$1'_'$2'_'$shot'shot_SAM_fixedval_lr_'$lr'_'$seed --lr $lr --source 1 --SAM --seed $seed &

                CUDA_VISIBLE_DEVICES=$5 python target_finetune.py --dataset $1 \
                --pretrain SHOT_LPFT --adapt $2 --few_shot $shot \
                --work_dir 'SHOT_LPFT_'$1'_'$2'_'$shot'shot_SAM_fixedval_lr_'$lr'_'$seed --lr $lr --source 2 --SAM --seed $seed &

                CUDA_VISIBLE_DEVICES=$6 python target_finetune.py --dataset $1 \
                --pretrain SHOT_LPFT --adapt $2 --few_shot $shot \
                --work_dir 'SHOT_LPFT_'$1'_'$2'_'$shot'shot_SAM_fixedval_lr_'$lr'_'$seed --lr $lr --source 3 --SAM --seed $seed &

                wait
            fi
        done
    done
done
# done
