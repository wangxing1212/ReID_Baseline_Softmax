import os
import torch as t

class BasicModule(t.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(t.load(path))

    def save(self, epoch_label):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        save_filename = self.model_name + '_%s.pth' % epoch_label
        save_dir = 'checkpoints/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir+'/'+save_filename )
        t.save(self.cuda().state_dict(),save_path)