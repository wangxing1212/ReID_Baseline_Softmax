import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import models
from tqdm import tqdm
from utils import re_ranking
from config import opt
import time

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def ranking(query_feature, gallery_feature, **kwargs):
    opt.parse(kwargs, show_config = False)
    
    if opt.re_ranking:
        print('Calculating initial distance')
        q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
        q_q_dist = np.dot(query_feature, np.transpose(query_feature))
        g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
        print('Re-ranking:')
        since = time.time()
        distance = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        time_elapsed = time.time() - since
        print('Reranking distance completed in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        ranking = np.zeros(distance.shape)
        print('Generating ranking list for each query image')
        for i in tqdm(range(distance.shape[0])):
            ranking[i]=np.argsort(distance[i])
    else:
        print('Calculating Euclidean Distance')
        distance = np.dot(query_feature, np.transpose(gallery_feature))
        ranking = np.zeros(distance.shape)
        print('Generating ranking list for each query image')
        for i in tqdm(range(distance.shape[0])):
            ranking[i]=np.argsort(distance[i])[::-1]
    
    return ranking

def evaluate(ranking,ql,qc,gl,gc):
    CMC = torch.IntTensor(len(gl)).zero_()
    ap = 0.0
    print('Calculating CMC and mAP')
    for i in tqdm(range(len(ql))):
        index = ranking[i]

        query_index = np.asarray(np.where(np.asarray(gl)==ql[i])).flatten()
        camera_index = np.asarray(np.where(np.asarray(gc)==qc[i])).flatten()

        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index1 = np.asarray(np.where(np.asarray(gl)==-1)).flatten()
        junk_index2 = np.intersect1d(query_index, camera_index)
        junk_index = np.append(junk_index2, junk_index1) #.flatten())

        ap_tmp = 0.0
        cmc_tmp = torch.IntTensor(len(index)).zero_()
        if good_index.size==0:   # if empty
            cmc_tmp[0] = -1

        # remove junk_index
        mask = np.in1d(index, junk_index, invert=True)
        index = index[mask]

        # find good_index index
        ngood = len(good_index)
        mask = np.in1d(index, good_index)
        rows_good = np.asarray(np.where(mask==True))
        rows_good = rows_good.flatten()

        cmc_tmp[rows_good[0]:] = 1
        
        for i in range(ngood):
            d_recall = 1.0/ngood
            precision = (i+1)*1.0/(rows_good[i]+1)
            if rows_good[i]!=0:
                old_precision = i*1.0/rows_good[i]
            else:
                old_precision=1.0
            ap_tmp = ap_tmp + d_recall*(old_precision + precision)/2
        
        CMC = CMC + cmc_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC/len(ql) #average CMC
    print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(ql)))