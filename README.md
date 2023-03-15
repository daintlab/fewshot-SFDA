# Few-shot Fine-tuning is All You Need for Source-free Domain Adaptation

### Requirments
- python == 3.9.12
- torch == 1.12.1
- torchvision == 0.13.1
- numpy == 1.23.5
- scikit-learn == 1.2.0

### Prepare Datasets
For vanilla SFDA setting, please prepare datasets in ```<data_dir>``` as follows :
```
|---- data_dir/
|      |---- office_home/
|            |---- Art
|                 |---- class 0 / *.png
|                 |---- ...
|                 |---- class 64 / *.png
|            |---- Clipart
|                 |---- class 0 / *.png
|                 |---- ...
|                 |---- class 64 / *.png
```
- Available datasets : `office31`, `office_home`,`VISDA-C`,`VLCS`,`terra_incognita`


### Prepare Source Model
We follow the same training procedure from SHOT to generate source pretrained model. Please refer to the official repository of [SHOT](https://github.com/tim-learn/SHOT) for source pretraining. All source models we used in the experiments will be released in future.

<hr>

### Few-shot Target Adaptation under Vanilla SFDA setting

#### 3-shot fine-tuning(FT) on Office31 dataset
```
# Adapt source domain 0 (Amazon) to all possible target domains (DSLR, Webcam)

CUDA_VISIBLE_DEVICES=0 python target_finetune.py --dataset office31 \
--pretrain SHOT --adapt all --few_shot 3 --work_dir <work_dir> \
--lr 1e-05 --SAM --seed 0 --source 0 --ckpt_dir <Path to source model directory>
```

#### 3-shot LP-FT on Office31 dataset
- First, train linear classifier only(LP) by changing `--adapt all` to `--adapt cls`.
```
# Adapt source domain 0 (Amazon) to all possible target domains (DSLR, Webcam)

CUDA_VISIBLE_DEVICES=0 python target_finetune.py --dataset office31 \
--pretrain SHOT --adapt cls --few_shot 3 --work_dir <work_dir> \
--lr 1e-04 --SAM --seed 0 --source 0 --ckpt_dir <Path to source model directory>
```
- Next, train whole network(FT) starting from LP checkpoint by changing `--pretrain SHOT` to `--pretrain SHOT_LP` and passing the path to LP checkpoint(work_dir in the above command) to `ckpt_dir`
```
# Adapt source domain 0 (Amazon) to all possible target domains (DSLR, Webcam)

CUDA_VISIBLE_DEVICES=0 python target_finetune.py --dataset office31 \
--pretrain SHOT_LP --adapt all --few_shot 3 --work_dir <work_dir> \
--lr 1e-05 --SAM --seed 0 --source 0 --ckpt_dir <Path to LP trained directory>
```

#### Results
- Log files and checkpoints will be saved as follows :
```
|---- logs/
|     |---- office31/
|           |---- work_dir/source0
|                |---- target 1
|                     |---- train_log.json
|                     |---- last_ckpt.pth # Last model
|                     |---- test_best_ckpt.pth # Model that gives the highest test accuracy
|                     |---- val_best_ckpt.pth # Model that gives the lowest validation loss
|                |---- target 2
|                |---- config.json
|                |---- best_target.json # Aggregated performance of test best model
|                |---- last_target.json # Aggregated performance of the last model
|                |---- selected_target.json # Aggregated performance of validation best model
```
- Note that all the reported values are from validation best model.

<hr> 

### Learning rates used in FT and LP-FT
For both FT and the second stage of LP-FT, we used `1e-05` for all benchmark datasets. For the first stage of LP-FT, we used `1e-04` for all benchmark datsets. All the learning rates are selected through 1-shot validation.

