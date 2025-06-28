import numpy as np
import os
import torch
from test_data import test_dataset
from saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm,cal_f2m,cal_s2m,cal_e2m
# 改设备管理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = './Image/NLPR/testsetNL'

dataset_path_pre = './model1/model_vgg16_DANet/(model_vgg16_DANet) test_20500'
# gai检查路径是否存在
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Ground truth dataset path {dataset_path} not found.")
if not os.path.exists(dataset_path_pre):
    raise FileNotFoundError(f"Predicted saliency map path {dataset_path_pre} not found.")
test_datasets = ['']

for dataset in test_datasets:
    sal_root = dataset_path_pre  +'/'
    gt_root = dataset_path  +'/GT/'
    test_loader = test_dataset(sal_root, gt_root)
    mae,fm,sm,em,wfm= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm()
    f2m = cal_f2m(test_loader.size)
    s2m = cal_s2m()
    e2m = cal_e2m()
    for i in range(test_loader.size):
        print ('predicting for %d / %d' % ( i + 1, test_loader.size))
        sal, gt = test_loader.load_data()
        if sal.size != gt.size:
            x, y = gt.size
            sal = sal.resize((x, y))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0
        res = sal
        res = np.array(res)
        if res.max() == res.min():
            res = res/255
        else:
            res = (res - res.min()) / (res.max() - res.min())
        mae.update(res, gt)
        sm.update(res,gt)
        fm.update(res, gt)
        em.update(res,gt)
        wfm.update(res,gt)
        # 更新新增指标
        f2m.update(res, gt)
        s2m.update(res, gt)
        e2m.update(res, gt)


    MAE = mae.show()
    maxf,meanf,_,_ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    # 新增指标结果
    maxf2, changeable_fm2, precision2, recall2 = f2m.show()
    sm2_score = s2m.show()
    em2_score = e2m.show()
    print('dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}  F2: {:.4f} S2: {:.4f} E2: {:.4f}'.format(
            dataset, MAE, maxf, meanf, wfm, sm,em,  maxf2, sm2_score,em2_score))