import warnings
class DefaultConfig(object):
    data_dir = '/home/linshan/Datasets/'
    dataset_name = 'Market1501'
    batch_size = 32
    num_epochs = 60
    save_rate = 10
    model = 'ResNet50'
    load_model_path = None
    load_epoch_label = 60

def parse(self,kwargs):
    '''
    根据字典kwargs 更新 config参数
    '''
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self,k,v)

def show_config(self,kwargs):
            
    print('user config:')
    print('------------')
    for k,v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            if k != 'parse':
                print(k,': ',getattr(self,k))
    print('------------')


DefaultConfig.parse = parse
opt =DefaultConfig()
default = DefaultConfig()