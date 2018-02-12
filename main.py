import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import models
from datasets import download_dataset
from datasets import pytorch_prepare
from datasets import dataset
from config import opt
from features import extract_features
from features import save_features
from features import load_features
from evaluation import ranking
from evaluation import evaluate
from evaluation import get_id
from tqdm import tqdm
from visdom import Visdom
from utils import Visualizer
from utils import RandomErasing
vis = Visualizer(opt.env)

################
#Train
################
def train(**kwargs):
   # server.main()
    opt.parse(kwargs,show_config=True)

    (image_datasets,
     train_dataloaders,
     test_dataloaders,
     dataset_sizes,
     classes) = dataset()
        
    model = getattr(models, opt.model)(len(classes))
    model.cuda()
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), 
                          lr = opt.lr, 
                          momentum=opt.momentum, 
                          weight_decay=opt.weight_decay, 
                          nesterov=opt.nesterov)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, 
                                           step_size=opt.scheduler_step, 
                                           gamma=opt.scheduler_gamma)

    since = time.time()
            
    if vis.check_connection():
        initial_loss = {
                    'Train Loss':1.0,
                    'Train Acc':0.0,
                    'Val Loss':1.0,
                    'Val Acc':0.0            
                }
        vis.plot_combine_all('Loss',initial_loss)
    
    for epoch in range(opt.num_epochs):
        
        print('Epoch {}/{}'.format(epoch+1, opt.num_epochs))
        
        for phase in ['train','val']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model.train(True)
            else:
                model.train(False)
            
            running_loss = 0.0
            running_corrects = 0
            
            for data in train_dataloaders[phase]:
                images, id = data
                
                images = Variable(images.cuda())
                id = Variable(id.cuda())
                
                optimizer.zero_grad()
                
                outputs = model(images)
                _, preds = torch.max(outputs.data,1)
                loss = criterion(outputs, id)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == id.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                train_epoch_loss = epoch_loss
                train_epoch_acc = epoch_acc
            else:
                val_epoch_loss = epoch_loss
                val_epoch_acc = epoch_acc
                
                if (epoch+1)%opt.save_rate == 0:
                    model.save(epoch+1)

        epoch_loss = {
                'Train Loss':train_epoch_loss,
                'Train Acc':train_epoch_acc,
                'Val Loss':val_epoch_loss,
                'Val Acc':val_epoch_acc
        }
        
        if vis.check_connection():
            vis.plot_combine_all('Loss',epoch_loss)
            
        print('-'*10)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
            
################
#Test
################
def test(**kwargs):
    opt.parse(kwargs, show_config = True)
    
    (image_datasets,
     train_dataloaders,
     test_dataloaders,
     dataset_sizes,classes) = dataset()
    
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam,gallery_label = get_id(gallery_path)
    query_cam,query_label = get_id(query_path)
    
    model = getattr(models, opt.model)(len(classes))
    model.load(opt.load_epoch_label)
    # Remove the final fc layer and classifier layer
    model.model.fc = nn.Sequential()
    model.classifier = nn.Sequential()
    # Change to test mode
    model = model.eval()
    model = model.cuda()
    
    if opt.load_features:
        all_features = load_features()
    else:
        all_features = extract_features(model,test_dataloaders, opt.flip)
        save_features(all_features)
        
    query_feature = all_features['query']
    gallery_feature = all_features['gallery']
    
    print('-'*30)
    result = ranking(query_feature,gallery_feature)
    
    print('-'*30)
    evaluate(result,query_label,query_cam,gallery_label,gallery_cam)
    
if __name__=='__main__':
    import fire
    fire.Fire()
