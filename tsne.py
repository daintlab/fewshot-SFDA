from sklearn.manifold import TSNE
import seaborn as sns
import torch
from model import ERM,SHOT,IMGNET,SHOT_fe
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import numpy as np

dset = 'office_home'
## source model
# ckpts = {}
# ckpts['netF'] = f'/nas/home/tmddnjs3467/domain-generalization/SHOT/object/ckps/source_seed2020/uda/{dset}/R/source_F.pt'
# ckpts['netB'] = f'/nas/home/tmddnjs3467/domain-generalization/SHOT/object/ckps/source_seed2020/uda/{dset}/R/source_B.pt'
# ckpts['netC'] = f'/nas/home/tmddnjs3467/domain-generalization/SHOT/object/ckps/source_seed2020/uda/{dset}/R/source_C.pt'
# model2 = SHOT(ckpts=ckpts, dataset=dset).cuda()

## Target
# shot = 1
# ckpts = f'/nas/home/tmddnjs3467/domain-generalization/source_free_DA/D_fine_tuning_output/{dset}/SHOT_{dset}_cls_1shot_SAM_fixedval_lr_1e-4_0/source3/target2/ckpt.pth'
# ckpts = '/nas/home/tmddnjs3467/domain-generalization/source_free_DA/D_fine_tuning_output/office_home/visualize_SHOT_LPFT/source0/target3/ckpt_1000.pth'
# model = SHOT_fe(ckpts=ckpts, dataset=dset).cuda()

# ckpts = f'/nas/home/tmddnjs3467/domain-generalization/source_free_DA/D_fine_tuning_output/{dset}/SHOT_{dset}_all_1shot_SAM_lr_1e-5_fixedval/source3/target2/ckpt.pth'
# model3 = SHOT_fe(ckpts=ckpts, dataset=dset).cuda()

## IMGNET
# ckpts = None
ckpts = f'/nas/home/tmddnjs3467/domain-generalization/source_free_DA/D_fine_tuning_output/office_home/IMGNET_office_home_cls_1shot_SAM_fixedval_lr_1e-4_0/source3/target2/best_ckpt.pth'
# ckpts = '/nas/home/tmddnjs3467/domain-generalization/source_free_DA/D_fine_tuning_output/office_home/visualize_IMGNET/source0/target3/ckpt_1000.pth'
model = IMGNET(ckpts=ckpts, dataset=dset).cuda()

ckpts = f'/nas/home/tmddnjs3467/domain-generalization/source_free_DA/D_fine_tuning_output/office_home/IMGNET_office_home_all_1shot_SAM_fixedval_lr_1e-5_0/source3/target2/best_ckpt.pth'
model3 = IMGNET(ckpts=ckpts, dataset=dset).cuda()

tsne = TSNE(random_state=42, perplexity=50)


transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
# t-SNE 시각화 함수 정의
def plot_vecs_n_labels(v, labels, fname, title):
    fig = plt.figure(figsize = (10,10))
    plt.axis('off')
    sns.set_style('darkgrid')
    # sns.set_palette("bright") 
    sns.scatterplot(v[:,0], v[:,1], legend=False, hue=labels, palette=sns.color_palette())
    # plt.legend(['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'])
    plt.title(title)
    plt.savefig(fname)

for s in [1]:
    class_num = 10
    random_class = np.random.RandomState(seed=s).permutation(65)[:class_num]

    dataset = ImageFolder(f'/data2/domainbed/{dset}/Product', transform)
    idx = [i for i in range(len(dataset.targets)) if dataset.targets[i] in random_class]
    # idx = list(range(dataset.targets.index(10)-1))
    dataset = Subset(dataset, idx)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)


    total_pred = torch.tensor([])
    # total_pred2 = torch.tensor([])
    total_pred3 = torch.tensor([])

    total_y = torch.tensor([])
    # total_y2 = torch.tensor([])
    total_y3 = torch.tensor([])
    for x, y in loader:
        x = x.cuda()
        with torch.no_grad():
            # pred = model.infer(x)
            pred = model(x)
            # pred2 = model2(x)
            # pred3 = model3.infer(x)
            pred3 = model3(x)

            total_pred = torch.cat((total_pred,pred.cpu()))
            # total_pred2 = torch.cat((total_pred2,pred2.cpu()))
            total_pred3 = torch.cat((total_pred3,pred3.cpu()))
            total_y = torch.cat((total_y,y.cpu()))
            # total_y2 = torch.cat((total_y2,y.cpu()))
            total_y3 = torch.cat((total_y3,y.cpu()))

            # total_pred = torch.cat((pred,pred3))
        # 모델의 출력값을 tsne.fit_transform에 입력하기
    pred_tsne = tsne.fit_transform(total_pred.data)
    # pred_tsne2 = tsne.fit_transform(total_pred2.data)
    pred_tsne3 = tsne.fit_transform(total_pred3.data)
    # pred_tsne1 = pred_tsne[:len(pred)]
    # pred_tsne2 = pred_tsne[len(pred):]
    # t-SNE 시각화 함수 실행

    # plot_vecs_n_labels(pred_tsne, total_y, f'tsne_seed{s}_LP.png', title='Source LP')
    # plot_vecs_n_labels(pred_tsne2, total_y2, f'tsne_seed{s}_source.png', title='Source only')
    # plot_vecs_n_labels(pred_tsne3, total_y3, f'tsne_seed{s}_FT.png', title='Source FT')


    plot_vecs_n_labels(pred_tsne, total_y, f'tsne_IMGNET_seed{s}_LP.png', title='IamgeNet LP')
    plot_vecs_n_labels(pred_tsne3, total_y3, f'tsne_IMGNET_seed{s}_FT.png', title='ImageNet FT')
