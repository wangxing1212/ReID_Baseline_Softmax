import os
from evaluation import load_result
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import *
import numpy as np
from config import opt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import clear_output
from utils import file_selector
    
def show_result(fname,R,size):
    result,CMC,mAP = load_result()
    ranking = result['ranking']
    query_imgs_path = result['query_imgs_path']
    gallery_imgs_path = result['gallery_imgs_path']

    img_path = os.path.join(opt.data_dir,opt.dataset_name,'query',fname)
    
    query_index = np.where(query_imgs_path==img_path)

    index = query_index[0][0]

    query_label = query_imgs_path[index].split('/')[-1].split('_')[0]

    fig, img = plt.subplots(1,R+1,figsize=(size,size))
    for i in range(0,R+1):
        if i == 0:
            img[i].imshow(mpimg.imread(query_imgs_path[index]))
            img[i].set_title('Query Image \n ID:'+str(int(query_label)))
        else:    
            img[i].imshow(mpimg.imread(gallery_imgs_path[int(ranking[index][i-1])]))
            gallery_label = gallery_imgs_path[int(ranking[index][i-1])].split('/')[-1].split('_')[0]
            if i == 1:
                img[i].set_title('Gallery Images \n ID:'+str(int(gallery_label)))
            else:
                img[i].set_title('ID:'+str(int(gallery_label)))

            if(gallery_label== query_label):
                autoAxis = img[i].axis()
                rec = Rectangle((autoAxis[0]-0.7,autoAxis[2]-0.2),(autoAxis[1]-autoAxis[0])+1,(autoAxis[3]-autoAxis[2])+0.4,fill=False,lw=2,color='red')
                rec = img[i].add_patch(rec)
                rec.set_clip_on(False)
        img[i].axis('off')

    plt.show()
    

def input_result_demo():
    
    file = widgets.Button(description='Open Query Image')
    display(file)
        
    def click(d): 
        clear_output()
        display(file,show)
        file_path = file_selector(opt.data_dir).decode("utf-8")
        fname = str(file_path).split('/')[-1]
        interact(show_result, 
         fname = fname,
         R=widgets.IntSlider(value=5,min=0,max=20), 
         size = widgets.IntSlider(value=20,min=0,max=40)
        )

    file.on_click(click)