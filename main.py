import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import models
from reid_dataset import download_dataset
from reid_dataset import pytorch_prepare
from config import opt
from tqdm import tqdm
from visdom import Visdom
viz = Visdom()

def dataset_process(**kwargs):
    opt.parse(kwargs)
    print('-'*40)
    download_dataset(opt.dataset_name, opt.data_dir)
    pytorch_prepare(opt.dataset_name, opt.data_dir)
    print('-'*40)
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

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(opt.data_dir+'/'+opt.dataset_name+'/pytorch/train_all'),
                                              transforms.Compose(transform_train_list))
    image_datasets['val'] = datasets.ImageFolder(os.path.join(opt.data_dir+'/'+opt.dataset_name+'/pytorch/val'),
                                              transforms.Compose(transform_val_list))
    image_datasets['query'] = datasets.ImageFolder(os.path.join(opt.data_dir+'/'+ opt.dataset_name + '/pytorch/query'), 
                                                  transforms.Compose(transform_test_list))
    image_datasets['gallery'] = datasets.ImageFolder(os.path.join(opt.data_dir+'/'+ opt.dataset_name + '/pytorch/gallery'),
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

def train(**kwargs):

    image_datasets,train_dataloaders,test_dataloaders,dataset_sizes,classes = dataset_process()
    
    opt.parse(kwargs,show_config=True)
        
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
    
    vis_epoch = np.array([0])
    train_loss = np.array([1])
    train_acc = np.array([0])
    val_loss = np.array([1])
    val_acc = np.array([0])
    
    
    visdom_loss = viz.line(
        Y=np.column_stack((train_loss,train_acc,val_loss, val_acc)),
        X=np.column_stack((vis_epoch, vis_epoch, vis_epoch, vis_epoch)),
        opts=dict(
            legend=['train_loss', 'train_acc','val_loss','val_acc'],
            fillarea=False,
            showlegend=False,
            xlabel='Epochs',
            ylabel='Loss',
            title='Training Loss',
        ),
    )
    since = time.time()
    for epoch in range(opt.num_epochs):
        print('-'*10)
        print('Epoch {}/{}'.format(epoch+1, opt.num_epochs))
        
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        val_epoch_loss = 0.0
        val_epoch_loss = 0.0
        
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
                    model.save(opt.dataset_name,epoch+1)
        
        vis_epoch = np.array([epoch+1])
        train_loss = np.array([train_epoch_loss])
        train_acc = np.array([train_epoch_acc])
        val_loss = np.array([val_epoch_loss])
        val_acc = np.array([val_epoch_acc])
                
        viz.line(
        Y=np.column_stack((train_loss,train_acc,val_loss, val_acc)),
        X=np.column_stack((vis_epoch, vis_epoch, vis_epoch, vis_epoch)),
        win=visdom_loss,
        update='append'
        )
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
            

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders,query_gallery):
    features = torch.FloatTensor()
    count = 0
    for data in tqdm(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
        #print(count)
        ff = torch.FloatTensor(n,2048).zero_()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img) 
            f = outputs.data.cpu()
            #print(f.size())
            ff = ff+f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features,ff), 0)
    return features

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

def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf
    score = np.dot(gf,query)
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    #index = index[0:2000]
    # good index
    query_index = np.asarray(np.where(np.asarray(gl)==ql)).flatten()
    camera_index = np.asarray(np.where(np.asarray(gc)==qc)).flatten()

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.asarray(np.where(np.asarray(gl)==-1)).flatten()
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.asarray(np.where(mask==True))
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2
        
    return ap, cmc

def test(**kwargs):
    
    image_datasets,train_dataloaders,test_dataloaders,dataset_sizes,classes = dataset_process()
    
    opt.parse(kwargs, show_config = True)
    
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam,gallery_label = get_id(gallery_path)
    query_cam,query_label = get_id(query_path)
    
    model = getattr(models, opt.model)(len(classes))
    model.load(opt.dataset_name, opt.load_epoch_label)
    
    # Remove the final fc layer and classifier layer
    model.model.fc = nn.Sequential()
    model.classifier = nn.Sequential()
    
    # Change to test mode
    model = model.eval()
    model = model.cuda()
    
    # Extract feature
    print('---------------------------')
    print('Extracting query features')
    query_feature = extract_feature(model,test_dataloaders['query'],'query')
    query_feature = query_feature.numpy()
    
    print('---------------------------')
    print('Extracting gallery features')
    gallery_feature = extract_feature(model,test_dataloaders['gallery'],'gallery')
    gallery_feature = gallery_feature.numpy()
   
    
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    print('-----------------------')
    print('Calculating CMC and mAP')
    for i in tqdm(range(len(query_label))):
        ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
    
if __name__=='__main__':
    import fire
    fire.Fire()