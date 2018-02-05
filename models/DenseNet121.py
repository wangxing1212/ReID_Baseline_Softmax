import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchvision import models
from .BasicModule import BasicModule
from config import opt

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)
        
class DenseNet121(BasicModule):
    def __init__(self, class_num,**kwargs):
        opt.parse(kwargs)
        super(DenseNet121, self).__init__()
        self.model_name = 'densenet121'
        self.dataset_name = opt.dataset_name
        
        model_ft = models.densenet121(pretrained=True)
        # add pooling to the model
        # in the originial version, pooling is written in the forward function 
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))

        num_ftrs = model_ft.classifier.in_features
        self.num_ftrs = num_ftrs
        
        add_block = []
        num_bottleneck = 512
        add_block += [nn.Linear(num_ftrs, num_bottleneck)]  #For ResNet, it is 2048
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.fc = add_block
        self.model = model_ft

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.model.features(x)  
        x = x.view(x.size(0),-1)
        x = self.model.fc(x)
        x = self.classifier(x)
        return x