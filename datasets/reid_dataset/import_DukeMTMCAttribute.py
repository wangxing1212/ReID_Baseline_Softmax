import os
from .reiddataset_downloader import *
from .import_DukeMTMC import *
import scipy.io
        
def import_DukeMTMCAttribute(dataset_dir):
    dataset_name = 'DukeMTMCAttribute'
    train,test,query = import_DukeMTMC(dataset_dir)
    reiddataset_downloader(dataset_name,dataset_dir)
    label=['backpack',
           'bag',
           'hangbag',
           'boots',
           'gender',
           'hat',
           'shoes',
           'top',
           'downblack',
           'downwhite',
           'downred',
           'downgray',
           'downblue',
           'downgreen',
           'downbrown',
           'upblack',
           'upwhite',
           'upred',
           'uppurple',
           'upgray',
           'upblue',
           'upgreen',
           'upbrown']
    
    
    train_person_id = []
    for personid in train:
        train_person_id.append(personid)
    train_person_id.sort(key=int)

    test_person_id = []
    for personid in test:
        test_person_id.append(personid)
    test_person_id.sort(key=int)
    
    f = scipy.io.loadmat(os.path.join(dataset_dir,dataset_name,'duke_attribute.mat'))

    test_attribute = {}
    train_attribute = {}
    for test_train in range(len(f['duke_attribute'][0][0])):
        if test_train == 1:
            id_list_name = 'test_person_id'
            group_name = 'test_attribute'
        else:
            id_list_name = 'train_person_id'
            group_name = 'train_attribute'
        for attribute_id in range(len(f['duke_attribute'][0][0][test_train][0][0])):
            for person_id in range(len(f['duke_attribute'][0][0][test_train][0][0][attribute_id][0])):
                id = locals()[id_list_name][person_id]
                if id not in locals()[group_name]:
                    locals()[group_name][id]=[]
                locals()[group_name][id].append(f['duke_attribute'][0][0][test_train][0][0][attribute_id][0][person_id])
    #two zero appear in train '0370' '0679'
    zero_check=[]
    for id in train_attribute:
        if 0 in train_attribute[id]:
            zero_check.append(id)
    for i in range(len(zero_check)):
        train_attribute[zero_check[i]] = [1 if x==0 else x for x in train_attribute[zero_check[i]]]

    return train_attribute,test_attribute,label

def import_DukeMTMCAttribute_binary(dataset_dir):
	train_duke_attr, test_duke_attr,label = import_DukeMTMCAttribute(dataset_dir)
	for id in train_duke_attr:
		train_duke_attr[id][:] = [x - 1 for x in train_duke_attr[id]]
	for id in test_duke_attr:
		test_duke_attr[id][:] = [x - 1 for x in test_duke_attr[id]]
	return train_duke_attr, test_duke_attr,label
