for lr in 1e-5
do
    for shot in 1 3
    do
        for oda_seed in 2020
        do
            for seed in 0 1 2
            do  
                if [ $1 = 'office31' ]
                then

                    CUDA_VISIBLE_DEVICES=$4 python target_finetune.py --dataset $1 \
                    --pretrain $2 --adapt $3 --few_shot $shot \
                    --work_dir $2'_'$1'_'$3'_'$shot'shot_SAM_fixedval_lr_'$lr'_subset_'$oda_seed'_'$seed --lr $lr --source 0 --SAM --seed $seed --subset --oda_seed $oda_seed &

                    CUDA_VISIBLE_DEVICES=$5 python target_finetune.py --dataset $1 \
                    --pretrain $2 --adapt $3 --few_shot $shot \
                    --work_dir $2'_'$1'_'$3'_'$shot'shot_SAM_fixedval_lr_'$lr'_subset_'$oda_seed'_'$seed --lr $lr --source 1 --SAM --seed $seed --subset --oda_seed $oda_seed &

                    CUDA_VISIBLE_DEVICES=$6 python target_finetune.py --dataset $1 \
                    --pretrain $2 --adapt $3 --few_shot $shot \
                    --work_dir $2'_'$1'_'$3'_'$shot'shot_SAM_fixedval_lr_'$lr'_subset_'$oda_seed'_'$seed --lr $lr --source 2 --SAM --seed $seed --subset --oda_seed $oda_seed &

                    wait

                else
                    CUDA_VISIBLE_DEVICES=$4 python target_finetune.py --dataset $1 \
                    --pretrain $2 --adapt $3 --few_shot $shot \
                    --work_dir $2'_'$1'_'$3'_'$shot'shot_SAM_fixedval_lr_'$lr'_subset_'$oda_seed'_'$seed --lr $lr --source 0 --SAM --seed $seed --subset --oda_seed $oda_seed &

                    CUDA_VISIBLE_DEVICES=$5 python target_finetune.py --dataset $1 \
                    --pretrain $2 --adapt $3 --few_shot $shot \
                    --work_dir $2'_'$1'_'$3'_'$shot'shot_SAM_fixedval_lr_'$lr'_subset_'$oda_seed'_'$seed --lr $lr --source 1 --SAM --seed $seed --subset --oda_seed $oda_seed &

                    CUDA_VISIBLE_DEVICES=$6 python target_finetune.py --dataset $1 \
                    --pretrain $2 --adapt $3 --few_shot $shot \
                    --work_dir $2'_'$1'_'$3'_'$shot'shot_SAM_fixedval_lr_'$lr'_subset_'$oda_seed'_'$seed --lr $lr --source 2 --SAM --seed $seed --subset --oda_seed $oda_seed &

                    CUDA_VISIBLE_DEVICES=$7 python target_finetune.py --dataset $1 \
                    --pretrain $2 --adapt $3 --few_shot $shot \
                    --work_dir $2'_'$1'_'$3'_'$shot'shot_SAM_fixedval_lr_'$lr'_subset_'$oda_seed'_'$seed --lr $lr --source 3 --SAM --seed $seed --subset --oda_seed $oda_seed &

                    wait
                fi
            done
        done
    done
done