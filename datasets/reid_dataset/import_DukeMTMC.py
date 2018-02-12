import os
from .reiddataset_downloader import*
def import_DukeMTMC(dataset_dir):
    reiddataset_downloader('DukeMTMC',dataset_dir)
    dukemtmc_dir = os.path.join(dataset_dir, 'DukeMTMC')
    data_group = ['train','test','query']
    for group in data_group:
        if group == 'train':
            name_dir = os.path.join(dukemtmc_dir , 'bounding_box_train')
        elif group == 'test':
            name_dir = os.path.join(dukemtmc_dir, 'bounding_box_test')
        else:
            name_dir = os.path.join(dukemtmc_dir, 'query')
        file_list=os.listdir(name_dir)
        globals()[group]={}
        for name in file_list:
            if name[-3:]=='jpg':
                id = name.split('_')[0]
                if id not in globals()[group]:
                    globals()[group][id]=[]
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])            
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                cam_n = int(name.split('_')[1][1])-1
                globals()[group][id][cam_n].append(os.path.join(name_dir,name))
    return train,test,query