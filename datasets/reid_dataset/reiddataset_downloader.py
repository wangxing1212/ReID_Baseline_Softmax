from __future__ import print_function
import os
import zipfile
import shutil
import requests
import h5py
import numpy as np
from PIL import Image
import argparse
from .gdrive_downloader import gdrive_downloader

dataset = {
    'CUHK01': '153IzD3vyQ0PqxxanQRlP9l89F1S5Vr47',
    'CUHK02': '0B2FnquNgAXoneE5YamFXY3NjYWM',
    'CUHK03': '1BO4G9gbOTJgtYIB0VNyHQpZb8Lcn-05m',
    'VIPeR':  '0B2FnquNgAXonZzJPQUtrcWJWbWc',
    'Market1501': '0B2FnquNgAXonU3RTcE1jQlZ3X0E',
    'Market1501Attribute' : '1YMgni5oz-RPkyKHzOKnYRR2H3IRKdsHO',
    'DukeMTMC': '1qtFGJQ6eFu66Tt7WG85KBxtACSE8RBZ0',
    'DukeMTMCAttribute' : '1eilPJFnk_EHECKj2glU_ZLLO7eR3JIiO'
}

def reiddataset_downloader(data_name, data_dir):
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    
    def cuhk03_to_image(data_dir):
        CUHK03_dir = os.path.join(data_dir , 'CUHK03')
        f = h5py.File(os.path.join(CUHK03_dir,'cuhk-03.mat'))
        
        detected_labeled = ['detected','labeled']

        for data_type in detected_labeled:

            datatype_dir = os.path.join(CUHK03_dir, data_type)
            if not os.path.exists(datatype_dir):
                    os.makedirs(datatype_dir)

            for campair in range(len(f[data_type][0])):
                campair_dir = os.path.join(datatype_dir,'P%d'%(campair+1))
                cam1_dir = os.path.join(campair_dir,'cam1')
                cam2_dir = os.path.join(campair_dir,'cam2')

                if not os.path.exists(campair_dir):
                    os.makedirs(campair_dir)
                if not os.path.exists(cam1_dir):
                    os.makedirs(cam1_dir)
                if not os.path.exists(cam2_dir):
                    os.makedirs(cam2_dir)

                for img_no in range(f[f[data_type][0][campair]].shape[0]):
                    if img_no < 5:
                        cam_dir = 'cam1'
                    else:
                        cam_dir = 'cam2'
                    for person_id in range(f[f[data_type][0][campair]].shape[1]):
                        img = np.array(f[f[f[data_type][0][campair]][img_no][person_id]])
                        if img.shape[0] !=2:
                            img = np.transpose(img, (2,1,0))
                            im = Image.fromarray(img)
                            im.save(os.path.join(campair_dir, cam_dir, "%d-%d.jpg"%(person_id+1,img_no+1)))
                    
    
    data_dir_exist = os.path.join(data_dir , data_name)
    
    if not os.path.exists(data_dir_exist):
        temp_dir = os.path.join(data_dir , 'temp')
        
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        destination = os.path.join(temp_dir , data_name)
        
        id = dataset[data_name]

        print("Downloading %s" % data_name)
        gdrive_downloader(id,destination)

        zip_ref = zipfile.ZipFile(destination)
        print("Extracting %s" % data_name)
        zip_ref.extractall(data_dir)
        zip_ref.close()
        shutil.rmtree(temp_dir)
        print("Done")
        if data_name == 'CUHK03':
            print('Converting cuhk03.mat into images')
            cuhk03_to_image(data_dir)
            print('Done')
    else:
        print("Dataset Check Success: %s exists!" %data_name)

def reiddataset_downloader_all(data_dir):
    for k,v in dataset.items():
        reiddataset_downloader(k,data_dir)

#For United Testing and External Use
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Name and Dataset Directory')
    parser.add_argument(dest="data_name", action="store", type=str,help="")
    parser.add_argument(dest="data_dir", action="store", default="~/Datasets/",help="")
    args = parser.parse_args() 
    reiddataset_downloader(args.data_name, args.data_dir)