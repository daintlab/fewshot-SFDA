import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
path = '/nas/datahub/domainbed/terra_incognita'

domain_dict = {'PACS': ['art_painting','cartoon','photo','sketch'],
                'office_home':['Art','Clipart','Product','Real_World'],
                'VLCS':['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
                'terra_incognita':['location_100', 'location_38', 'location_43', 'location_46'],
                'office31':['amazon', 'dslr', 'webcam']}

domain = path.split('/')[-1]
class_dict = defaultdict(list)
for d in domain_dict[domain]:
    dirs = os.listdir(path+'/'+d)
    for c in dirs:
        files = [i for i in os.listdir(path+'/'+d+'/'+c) if i[-4:]=='.jpg' or '.png']
        class_dict[c] = len(files)
    items = class_dict.items()
    x, y = zip(*items)
    y = [round(i * 0.8, 1) for i in y]
    lower = [i for i in range(len(y)) if y[i] < 12] # 10-shot FT를 하려면 validation까지 class당 최소 11장의 image는 있어야함
    x = [f'{i}({j})' for i,j in zip(x,y)]
    plt.figure(figsize=(10,8))
    plt.plot(x,y,'--o')
    plt.title(f'{domain.upper()} - {d.upper()}')
    plt.xlabel('class name')
    plt.ylabel('number of images')
    plt.xticks(rotation=45)
    plt.axhline(10, color='red', linestyle='--')
    plt.show()
    if not os.path.exists(f'./vis/{domain}'):
        os.mkdir(f'./vis/{domain}')
    plt.savefig(f'./vis/{domain}/{d}.png')
    plt.close()
    [print(f'{d} 도메인의 이미지 수가 적은 것, {x[i]}') for i in lower] # class당 11장이 안되는 class를 print
