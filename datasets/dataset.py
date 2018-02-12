import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import os
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

from datasets import reiddataset_downloader
from datasets import pytorch_prepare
from config import opt

from utils import RandomErasing

def dataset(**kwargs):
    opt.parse(kwargs)
    print('-'*40)
    reiddataset_downloader(opt.dataset_name, opt.data_dir)
    pytorch_prepare(opt.dataset_name, opt.data_dir)
    print('-'*40)
    
    train_all = ''
    if opt.train_all:
        train_all = '_all'

    transform_train_list = [
            transforms.Resize(144),
            transforms.RandomCrop((256,128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    transform_val_list = [
            transforms.Resize(size=(256,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    transform_test_list =[
            transforms.Resize((288,144), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
    
    if opt.random_erasing_p > 0:
        transform_train_list = transform_train_list + [RandomErasing(opt.random_erasing_p)]
        transform_val_list = transform_val_list + [RandomErasing(opt.random_erasing_p)]

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(opt.data_dir,opt.dataset_name,'pytorch','train'+train_all),
                                              transforms.Compose(transform_train_list))
    image_datasets['val'] = datasets.ImageFolder(os.path.join(opt.data_dir,opt.dataset_name,'pytorch','val'),
                                              transforms.Compose(transform_val_list))
    image_datasets['query'] = datasets.ImageFolder(os.path.join(opt.data_dir,opt.dataset_name , 'pytorch','query'), 
                                                  transforms.Compose(transform_test_list))
    image_datasets['gallery'] = datasets.ImageFolder(os.path.join(opt.data_dir, opt.dataset_name , 'pytorch','gallery'),
                                                  transforms.Compose(transform_test_list))
    train_dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch_size,
                                                    shuffle=True, num_workers=4)
                                                  for x in ['train', 'val']}
    test_dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch_size,
                                                    shuffle=False, num_workers=4)
                                                  for x in ['query','gallery']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','query','gallery']}
    classes = image_datasets['train'].classes  
    
    return image_datasets,train_dataloaders,test_dataloaders,dataset_sizes,classes