import warnings
warnings.filterwarnings('ignore','.*conversion.*')

import os
import h5py
from PIL import  Image
from torch.utils import data
import numpy as np
from torchvision import  transforms as T
from config import opt
from utils import RandomErasing

class Train_Dataset_HDF5(data.Dataset):
    def __init__(self,transforms = None, train_val = 'train', **kwargs):
        opt.parse(kwargs,show_config=False)

        self.hdf5_path = os.path.join(opt.data_dir,opt.dataset_name,opt.dataset_name+'.hdf5')
        self.train_val = train_val
        
        dataset = h5py.File(self.hdf5_path,'r')
        
        names = list(dataset['train'].keys())
        val_ids=[]
        val_names=[]
        for name in names:
            id = name.split('_')[0]
            if id not in val_ids:
                val_ids.append(id)
                val_names.append(name)

        if self.train_val == 'train':
            self.train_names = names
        elif self.train_val == 'val':
            self.train_names = val_names
        else:
            print('Input should only be train or val')
            
        self.num_ids = len(dataset['ids']['train'])
        dataset.close()
        
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
        name = self.train_names[index]
        dataset = h5py.File(self.hdf5_path,'r')
        
        i = dataset['train'][name]['index'].value   
        id = dataset['train'][name]['id'].value

        data = Image.fromarray(np.uint8(dataset['train'][name]['img']))
        data = self.transforms(data)
        return data, i, id
    
    def __len__(self):
        return len(self.train_names)
        
    def num_ids(self):
        return self.num_ids
    
class Test_Dataset_HDF5(data.Dataset):
    def __init__(self, transforms = None, query_gallery='query', **kwargs):
        opt.parse(kwargs,show_config=False)
        self.hdf5_path = os.path.join(opt.data_dir,opt.dataset_name,opt.dataset_name+'.hdf5')
        self.query_gallery = query_gallery
        
        dataset = h5py.File(self.hdf5_path,'r')
        query_names = list(dataset['query'].keys())
        gallery_names = list(dataset['gallery'].keys())

        if query_gallery == 'query':
            self.test_names = query_names
        elif query_gallery == 'gallery':
            self.test_names = gallery_names
        else:
            print('Input shoud only be query or gallery;')
        dataset.close()
        
        if transforms is None:
            self.transforms = T.Compose( [
                    T.Resize(size=(256,128)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                
    def __getitem__(self,index):
        name = self.test_names[index]
        f = h5py.File(self.hdf5_path,'r')
        
        if self.query_gallery == 'query':
            dataset = f['query']
        elif query_gallery == 'gallery':
            dataset = f['gallery']
        else:
            print('Input shoud only be query or gallery;')
        
        
        id = dataset[name]['id'].value
        i = dataset[name]['index'].value

        data = Image.fromarray(np.uint8(dataset[name]['img']))
        data = self.transforms(data)
        return data, i, id
    
    def __len__(self):
        return len(self.test_names)