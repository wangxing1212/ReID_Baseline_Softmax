import os
from PIL import  Image
from torch.utils import data
import numpy as np
from torchvision import  transforms as T
from .reid_dataset import market_duke
from config import opt
from utils import RandomErasing

class Train_Dataset(data.Dataset):

    def __init__(self,transforms = None, train_val = 'train', **kwargs):
        opt.parse(kwargs,show_config=False)
        
        train,query,gallery = market_duke(opt.data_dir,opt.dataset_name)

        if train_val == 'train':
            self.train_data = train['data']
            self.train_ids = train['ids']
        elif train_val == 'val':
            val_data = {'data':[],'ids':[]}
            for path,id,cam_n in train['data']:
                if id not in val_data['ids']:
                    val_data['data'].append(path)
                    val_data['ids'].append(id)
            self.train_data = val_data['data']
            self.train_ids = val_data['ids']
        else:
            print('Input should only be train or val')
        
        self.num_ids = len(self.train_ids)
            
        if transforms is None:
            if train_val == 'train':
                if opt.random_erasing_p > 0:
                    self.transforms = T.Compose([
                                T.Resize(144),
                                T.RandomCrop((256,128)),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                RandomErasing(opt.random_erasing_p),
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
                else:
                    self.transforms = T.Compose([
                                T.Resize(144),
                                T.RandomCrop((256,128)),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
            else:
                    self.transforms = T.Compose( [
                            T.Resize(size=(256,128)),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
                
        
    def __getitem__(self,index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.train_data[index][0]
        id = self.train_data[index][1]
        label = self.train_ids.index(id)
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label, id
    
    def __len__(self):
        return len(self.train_data)
    
    def num_ids(self):
        return self.num_ids
    
    
class Test_Dataset(data.Dataset):
    def __init__(self, transforms = None, query_gallery='query', **kwargs):
        opt.parse(kwargs,show_config=False)
        train,query,gallery = market_duke(opt.data_dir,opt.dataset_name)
        
        if query_gallery == 'query':
            self.test_data = query['data']
            self.test_ids = query['ids']
        elif query_gallery == 'gallery':
            self.test_data = gallery['data']
            self.test_ids = gallery['ids']
        else:
            print('Input shoud only be query or gallery;')
        
        if transforms is None:
            self.transforms = T.Compose( [
                    T.Resize(size=(256,128)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                
    def __getitem__(self,index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.test_data[index][0]
        id = self.test_data[index][1]
        cam_n = self.test_data[index][2]
        label = self.test_ids.index(id)
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label, id, cam_n
    
    def __len__(self):
        return len(self.test_data)