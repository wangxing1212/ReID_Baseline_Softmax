import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
from config import opt
import models
from utils import check_jupyter_run
if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_features(model, dataloaders, flip = True):
    num_ftrs = model.num_ftrs
    all_features = dict.fromkeys(dataloaders.keys())
    for k,v in dataloaders.items():
        features = torch.FloatTensor()
        id_list = []
        cam_list = []
        name_list = []
        print('Extracting '+ k +' features')
        for data in tqdm(v):
            images, indices, ids, cams, names = data
            n, c, h, w = images.size()
            ff = torch.FloatTensor(n,num_ftrs).zero_()
            if flip:
                for i in range(2):
                    if(i==1):
                        images = fliplr(images)
                    input_img = Variable(images.cuda())
                    outputs = model(input_img) 
                    f = outputs.data.cpu()
                    #print(f.size())
                    ff = ff+f
            else:
                input_img = Variable(images.cuda())
                outputs = model(input_img) 
                ff = outputs.data.cpu()
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features,ff), 0)
            name_list = name_list + list(names)
            id_list = id_list + list(ids)            
            cam_list = cam_list + list(cams.numpy())
            
        all_features[k]=[features.numpy(),id_list,cam_list,name_list]
        
    return all_features

def save_features(all_features, **kwargs):
    opt.parse(kwargs, show_config = False)

    save_filename = (opt.dataset_name+'_'+opt.model + '_epo%s' % opt.load_epoch_label)
    if opt.annotation != None:
        save_dir = os.path.join('features',
                                opt.dataset_name,
                                opt.model,
                                opt.annotation)
    else:
        save_dir = os.path.join('features',
                                opt.dataset_name,
                                opt.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir,save_filename )
        
    np.save(save_path, all_features)
    print('Features saved to '+save_path)

def load_features(**kwargs):
    opt.parse(kwargs, show_config = False)        
    load_filename = (opt.dataset_name+'_'+
                     opt.model + '_epo%s.npy' % opt.load_epoch_label)
    if opt.annotation != None:
        load_dir = os.path.join('features',
                                opt.dataset_name,
                                opt.model,
                                opt.annotation)
    else:
        load_dir = os.path.join('features',
                                opt.dataset_name,
                                opt.model)
    if not os.path.exists(load_dir):
        print('Features are not existed, please do feature extraction first') 
    features = np.load(os.path.join(load_dir,load_filename))
    print('Features Load successfully')
    return features.item()

