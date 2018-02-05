import os
import torch as t

class BasicModule(t.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, epoch_label):
        '''
        可加载指定路径的模型
        '''
        save_filename = (self.model_name 
                         + '_epo%s.pth' % epoch_label)
        save_path = os.path.join('./checkpoints/'
                                 +self.dataset_name+'/'
                                 +self.model_name+'/'
                                 +save_filename)
        self.load_state_dict(t.load(save_path))
        print(save_path)

    def save(self, epoch_label):
        '''
        保存模型，默认使用“模型名字+Epoche”作为文件名
        '''
        save_filename = (self.model_name 
                         + '_epo%s.pth' % epoch_label)
        save_dir = 'checkpoints/'+self.dataset_name+'/'+self.model_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir+'/'+save_filename )
        t.save(self.cuda().state_dict(),save_path)
    
    def num_ftrs(self):
        return self.num_ftrs