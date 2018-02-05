import visdom
import time
import numpy as np

class Visualizer(object):
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    '''

    def __init__(self, env='main', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        
        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {} 
        self.log_text = ''
        
    def reinit(self,env='main',**kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env,**kwargs)
        return self

    def plot(self, name, y,**kwargs):
        '''
        self.plot('loss',1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1
        
    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        '''
        for k, v in d.items():
            self.plot(k, v)
            
    def plot_combine_many(self, name, d):
        x = self.index.get(name, 0)
        X = []
        Y = []
        legend = []
        for k, v in sorted(d.items()):
            Y.append(v)
            X.append(x)
            legend.append(k)
        Y = np.array([Y])
        X = np.array([X])
        self.vis.line(
            Y=Y, 
            X=X,
            win=name,
            opts=dict(
                title=name,
                legend=legend
            ),
            update=None if x == 0 else 'append',
        )
        self.index[name] = x + 1
            

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)


    def img(self, name, img_,**kwargs):
        '''
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)

        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        '''
        self.vis.images(img_.cpu().numpy(),
                       win=name,
                       opts=dict(title=name),
                       **kwargs
                       )


    def log(self,info,win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
                            time=time.strftime('%m%d_%H%M%S'),\
                            info=info)) 
        self.vis.text(self.log_text,win)   

    def __getattr__(self, name):
        return getattr(self.vis, name)