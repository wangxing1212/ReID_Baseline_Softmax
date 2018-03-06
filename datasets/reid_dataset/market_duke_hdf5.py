import warnings
warnings.filterwarnings('ignore','.*conversion.*')

import os
import h5py
import numpy as np
from PIL import  Image
from reid_dataset import market_duke

def market_duke_hdf5(data_dir,dataset_name,save_dir=os.getcwd()):
    phase_list = ['train','query','gallery']
    dataset = market_duke(data_dir,dataset_name)
    f = h5py.File(os.path.join(save_dir,dataset_name+'.hdf5'),'w')
    for phase in phase_list:
        grp = f.create_group(phase)
        phase_dataset = dataset[phase_list.index(phase)]
        for i in range(len(phase_dataset['data'])):
            name = phase_dataset['data'][i][0].split('/')[-1].split('.')[0]
            temp = grp.create_group(name) 
            temp.create_dataset('img',data=Image.open(phase_dataset['data'][i][0]))
            temp.create_dataset('id',data=int(phase_dataset['data'][i][1]))
            temp.create_dataset('cam',data=int(phase_dataset['data'][i][2]))