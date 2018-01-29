import warnings
class DefaultConfig(object):
    data_dir = '/home/linshan/Dataset/'
    batch_size = 32
    num_epochs = 50
    save_dir = '/home/linshan/ResearchProjects/reid_baseline/checkpoints/'
    save_rate = 1
    model = 'ResNet50'

def parse(self,kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k,getattr(self,k))


DefaultConfig.parse = parse
opt =DefaultConfig()
default = DefaultConfig()