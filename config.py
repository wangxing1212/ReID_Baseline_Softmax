import warnings
class DefaultConfig(object):
    data_dir = './datasets/'
    #data_dir = '/home/linshan/Datasets'
    dataset_name = 'Market1501'
    batch_size = 32
    num_epochs = 50
    save_rate = 10
    model = 'ResNet50'
    load_epoch_label = 50
    lr =0.01
    momentum = 0.9
    weight_decay = 5e-4
    nesterov = True
    scheduler_step = 40
    scheduler_gamma = 0.1
    flip = True
    load_features = False
    re_ranking = True
    random_erasing_p = 0
    annotation = None
    hdf5 = True
    num_workers = 4

def parse(self,kwargs,show_config=False):
    '''
    根据字典kwargs 更新 config参数
    '''
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self,k,v)
    
    if show_config == True:
        print('Current Configuration:')
        for k,v in sorted(self.__class__.__dict__.items()):
            if not k.startswith('__'):
                if k != 'parse':
                    print(k,': ',getattr(self,k))
        print('-'*30)

DefaultConfig.parse = parse
opt =DefaultConfig()
default = DefaultConfig()