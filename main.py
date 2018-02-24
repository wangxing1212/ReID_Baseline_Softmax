import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import models
from datasets import dataset
from config import opt
from features import extract_features
from features import save_features
from features import load_features
from evaluation import ranking
from evaluation import evaluate
from evaluation import save_result
from visdom import Visdom
from utils import Visualizer
from utils import RandomErasing
from datasets import Train_Dataset
from datasets import Test_Dataset
from utils import check_jupyter_run
if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
vis = Visualizer()
num_classes = Train_Dataset(train_val = 'train').num_ids
################
#Train
################
def train(**kwargs):
    opt.parse(kwargs,show_config=True)
    
    train_dataloaders = {x: DataLoader(Train_Dataset(train_val = x),
                                       batch_size=opt.batch_size,
                                       shuffle=True, num_workers=4)
                         for x in ['train', 'val']}
    
    dataset_sizes = {x:len(Train_Dataset(train_val = x)) for x in ['train', 'val']}
        
    model = getattr(models, opt.model)(num_classes)
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
    
    print('-'*40)
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
            
            for data in tqdm(train_dataloaders[phase]):
                images, labels, ids, img_path = data
                
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
                
                optimizer.zero_grad()
                
                outputs = model(images)
                _, preds = torch.max(outputs.data,1)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                
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
    
    test_dataloaders = {x: DataLoader(Test_Dataset(query_gallery = x),
                                       batch_size=opt.batch_size,
                                       shuffle=False, num_workers=4)
                         for x in ['query', 'gallery']}
    
    model = getattr(models, opt.model)(num_classes)
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
        
    query_feature = all_features['query'][0]
    gallery_feature = all_features['gallery'][0]
    
    print('-'*30)
    result = ranking(query_feature,gallery_feature)
    
    print('-'*30)
    gallery_label = all_features['gallery'][1]
    gallery_cam = all_features['gallery'][2]
    gallery_imgs_path = all_features['gallery'][3]
    query_label = all_features['query'][1]
    query_cam = all_features['query'][2]
    query_imgs_path = all_features['query'][3]
    
    result,CMC,mAP = evaluate(result,query_label,query_cam,gallery_label,gallery_cam)
    save_result(result,query_imgs_path,gallery_imgs_path,CMC,mAP)
    
if __name__=='__main__':
    import fire
    fire.Fire()