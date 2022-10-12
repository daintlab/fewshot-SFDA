# source_free_DA

### Available dataset
- PACS
- OfficeHome
- DomainNet

<hr>

### Train on source data

#### Single-source training
- Train single model
```
CUDA_VISIBLE_DEVICES=0 python train.py --data-dir /data/domainbed --dataset <Dataset> --source <source domain idx>
```
- Train all models for each source domain
```
bash scripts/train_single_src.sh <Dataset Name> <first GPU idx> ...
```

#### Multi-source training
- Train single model
```
CUDA_VISIBLE_DEVICES=0 python train.py --data-dir /data/domainbed --dataset <Dataset> --target <target domain idx>
```
- model will be trained on all domains except ```--target``` domain
- Train all models for each source domains
```
bash scripts/train_multi_src.sh <Dataset Name> <first GPU idx> ...
```

#### Checkpoints are available at ```/nas/home/sangwoo/SFDA/ckpts```
- PACS(single source, multi source)
- OfficeHome(single source, multi source)
- DomainNet(single source, multi source)

<hr>

### Adapt
- adapt source-trained model on target domain
- adaptation dataset is equal to test dataset(TODO : split)
- adapt mode(```--adapt```)
  - ```None```(no adapt) 
  - ```stat```(stat only)
  - ```affine-scale```(train scale param.), ```affine-shift```(train shift param.), ```affine```(train affine param.)
  - ```BN```(train stat&affine) 
- command
```
CUDA_VISIBLE_DEVICES=0 python adapt.py --ckpt <path to source trained model> \
--target <target domain idx> --adapt <adapt mode>
```
