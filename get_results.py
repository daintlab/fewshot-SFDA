import pandas as pd
import os
import json
import numpy as np

def select_by_val_loss(path):
    target_list = [path+'/'+i for i in os.listdir(path) if os.path.isdir(path+'/'+i)]
    val_loss = []
    for i in target_list:
        json_data = []
        with open(f'{i}/train_log.json', 'r') as f:
            for line in f:
                json_data.append(json.loads(line))
            json_data = json_data[-11:]
        loss = [i['val_loss'] for i in json_data]
        assert len(loss) == 11
        val_loss.append(np.min(loss))
    return np.mean(val_loss)

# get fine-tuning result
dataset = 'office31'
root_dir = f'./D_fine_tuning_output/{dataset}'

## 여러가지 configuraion의 range 설정
n_shot = [3]
methods = ['clsBN']
lrs = ['1e-4']
# mixup = [1,5,10,15]

df = pd.DataFrame()
for mode in ['best']: # best, last, seleted 지정 가능
    for shot in n_shot:
        for method in methods:
            for lr in lrs:
                for seed in [0,1,2]:
                    # for wd in ['1e-1','1e-2','1e-3','1e-4','1e-5']:
                    if seed == 0:
                        name = '_0'
                    elif seed == 1:
                        name = '_1'
                    elif seed == 2:
                        name = '_2'
                    
                    # 저장된 폴더 이름 설정
                    root = os.path.join(root_dir,f'SHOT_{dataset}_{method}_{shot}shot_SAM_lr_{lr}_fixedval'+name)
                    # root = os.path.join(root_dir,f'SHOT_{dataset}_{method}_{shot}shot_SAM_lastdance')
                    # root = os.path.join(root_dir,f'SHOT_{dataset}_{method}_{shot}shot_SAM_fixedval_lr_{lr}_robust'+name)
                    # root = os.path.join(root_dir,f'SHOT_{dataset}_{method}_{shot}shot_mixup20_lr_{lr}')
                    # root = os.path.join(root_dir,f'SHOT_{dataset}_{method}_{shot}shot_SAM_regmixup{mix}_lr_{lr}')
            
                    mode_result = {}
                    mode_result['shot'] = shot
                    mode_result['method'] = method
                    mode_result['mode'] = mode
                    mode_result['lr'] = lr
                    # mode_result['wd'] = wd
                    mode_result['seed'] = seed
                    # mode_result['mix'] = mix

                    if dataset == 'office31':
                        num_d = 3
                    else:
                        num_d = 4
                    loss_sum = 0
                    for source in range(num_d):
                        path = os.path.join(root,f'source{source}/{mode}_target.json')
                        # validation loss 계산
                        loss_sum += select_by_val_loss(os.path.join(root,f'source{source}'))
                        with open(path,'r') as f:
                            result = json.load(f)
                        mode_result.update(result)
                    mode_result['sum_of_min_val_loss'] = loss_sum
                    df = df.append(mode_result,ignore_index=True)
df = df.round(2)
df['avg'] = df.iloc[:,5:-1].apply(np.mean, axis=1)
final_df = df.groupby(['shot','method','mode','lr']).mean()[['avg','sum_of_min_val_loss']]
final_df['std'] = df.groupby(['shot','method','mode','lr']).std()['avg']

final_df = final_df.sort_values(by=['shot','mode','lr'])
final_df.to_csv('./result.csv')
