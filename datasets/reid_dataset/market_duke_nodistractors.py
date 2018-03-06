import os
from .reiddataset_downloader import *
def market_duke_nodistractors(data_dir, dataset_name):
    dataset_dir = os.path.join(data_dir,dataset_name)
    
    if not os.path.exists(dataset_dir):
        reiddataset_downloader(dataset_name,data_dir)
        
    dataset_dir = os.path.join(data_dir,dataset_name)
    data_group = ['train','query','gallery']
    for group in data_group:
        if group == 'train':
            name_dir = os.path.join(dataset_dir , 'bounding_box_train')
        elif group == 'query':
            name_dir = os.path.join(dataset_dir, 'query')
        else:
            name_dir = os.path.join(dataset_dir, 'bounding_box_test')
        file_list=sorted(os.listdir(name_dir))
        globals()[group]={}
        globals()[group]['data']=[]
        globals()[group]['ids'] = []
        for name in file_list:
            if name[-3:]=='jpg':
                id = name.split('_')[0]
                if (id!='0000' and id !='-1'):
                    cam_n = int(name.split('_')[1][1])
                    images = os.path.join(name_dir,name)
                    globals()[group]['data'].append([images,id,cam_n])
                    if id not in globals()[group]['ids']:
                        globals()[group]['ids'].append(id)
    return train,query,gallery