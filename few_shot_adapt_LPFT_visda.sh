for lr in 1e-5
do
    for shot in 10 20 30
    do
        for seed in 0 1 2
        do
            if [ $1 = 'VISDA-C' ]
            then

                CUDA_VISIBLE_DEVICES=$3 python target_finetune.py --dataset $1 \
                --pretrain SHOT_LPFT --adapt $2 --few_shot $shot \
                --work_dir 'SHOT_LPFT_'$1'_'$2'_'$shot'shot_SAM_fixedval_lr_'$lr'_'$seed --lr $lr --source 0 --SAM --seed $seed &
                
                wait
            fi
        done
    done
done
# done