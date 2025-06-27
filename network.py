import torch
from torch import nn
import numpy as np

import timm

class NFRES50(nn.Module) :
    def __init__(self,cls_nums):
        super (NFRES50,self).__init__()
        self.cls_nums = cls_nums
        self.fe = timm.create_model('nf_resnet50', pretrained=True, num_classes=0, global_pool='')
        self.id = nn.Identity()
        self.relu = nn.ReLU()
        self.header = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048,self.cls_nums),
            nn.Identity()
        )
    def forward(self,inputs):
        A = self.fe(inputs)
        A = self.id(A)
        A = self.relu(A)
        O = self.header(A)
        return O
    
class RESNEXT26(nn.Module) :
    def __init__(self,cls_nums):
        super (RESNEXT26,self).__init__()
        self.cls_nums = cls_nums
        self.fe = timm.create_model('bat_resnext26ts', pretrained=True, num_classes=0, global_pool='')
        self.id = nn.Identity()
        self.relu = nn.ReLU()
        self.header = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048,self.cls_nums),
            nn.Identity()
        )
    def forward(self,inputs):
        A = self.fe(inputs)
        A = self.id(A)
        A = self.relu(A)
        O = self.header(A)
        return O
    
class BOTNET26t(nn.Module) :
    def __init__(self,cls_nums):
        super (BOTNET26t,self).__init__()
        self.cls_nums = cls_nums
        self.fe = timm.create_model('botnet26t_256', pretrained=True, num_classes=0, global_pool='')
        self.id = nn.Identity()
        self.relu = nn.ReLU()
        self.header = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048,self.cls_nums),
            nn.Identity()
        )
    def forward(self,inputs):
        A = self.fe(inputs)
        A = self.id(A)
        A = self.relu(A)
        O = self.header(A)
        return O
    
class REXNET100(nn.Module) :
    def __init__(self,cls_nums):
        super (REXNET100,self).__init__()
        self.cls_nums = cls_nums
        self.fe = timm.create_model('rexnet_100', pretrained=True, num_classes=0, global_pool='')
        self.id = nn.Identity()
        self.relu = nn.ReLU()
        self.header = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1280,self.cls_nums),
            nn.Identity()
        )
    def forward(self,inputs):
        A = self.fe(inputs)
        A = self.id(A)
        A = self.relu(A)
        O = self.header(A)
        return O
    
class MOBILENET_SMALL(nn.Module) :
    def __init__(self,cls_nums):
        super (MOBILENET_SMALL,self).__init__()
        self.cls_nums = cls_nums
        self.fe = timm.create_model('mobilenetv3_small_050', pretrained=True, num_classes=0, global_pool='')
        self.id = nn.Identity()
        self.relu = nn.ReLU()
        self.header = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024,self.cls_nums),
            nn.Identity()
        )
    def forward(self,inputs):
        A = self.fe(inputs)
        A = self.id(A)
        A = self.relu(A)
        O = self.header(A)
        return O
    
class ReshapeOutput(nn.Module):
    def __init__(self):
        super(ReshapeOutput, self).__init__()
    
    def forward(self, x):
        # 添加一个新的维度，使形状从 (-1, 1, 384) 变为 (-1, 1, 1, 384)
        x = x[:, 0:1, :]
        return x.unsqueeze(2).permute(0, 3, 1, 2)
    
class Vit(nn.Module) :
    def __init__(self,cls_nums):
        super (Vit,self).__init__()
        self.cls_nums = cls_nums
        self.fe = timm.create_model('vit_small_r26_s32_224', pretrained=True, num_classes=0, global_pool='')
        # self.id = nn.Identity()
        self.relu = nn.ReLU()
        self.reshape = ReshapeOutput()
        
        self.header = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(384,self.cls_nums),
            nn.Identity()
        )
    def forward(self,inputs):
        A = self.fe(inputs)
        # A = self.id(A)
        A = self.relu(A)
        A = self.reshape(A)
        
        O = self.header(A)
        return O
    
class Cait_Small(nn.Module) :
    def __init__(self,cls_nums):
        super (Cait_Small,self).__init__()
        self.cls_nums = cls_nums
        self.fe = timm.create_model('cait_s24_224', pretrained=True, num_classes=0, global_pool='')
        # self.id = nn.Identity()
        self.relu = nn.ReLU()
        self.reshape = ReshapeOutput()
        
        self.header = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(384,self.cls_nums),
            nn.Identity()
        )
    def forward(self,inputs):
        A = self.fe(inputs)
        # A = self.id(A)
        A = self.relu(A)
        A = self.reshape(A)
        
        O = self.header(A)
        return O
    
class Convit_Tiny(nn.Module) :
    def __init__(self,cls_nums):
        super (Convit_Tiny,self).__init__()
        self.cls_nums = cls_nums
        self.fe = timm.create_model('convit_tiny', pretrained=True, num_classes=0, global_pool='')
        # self.id = nn.Identity()
        self.relu = nn.ReLU()
        self.reshape = ReshapeOutput()
        
        self.header = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(192,self.cls_nums),
            nn.Identity()
        )
    def forward(self,inputs):
        A = self.fe(inputs)
        # A = self.id(A)
        A = self.relu(A)
        A = self.reshape(A)
        
        O = self.header(A)
        return O

class Convit_Base(nn.Module) :
    def __init__(self,cls_nums):
        super (Convit_Base,self).__init__()
        self.cls_nums = cls_nums
        self.fe = timm.create_model('convit_base', pretrained=True, num_classes=0, global_pool='')
        # self.id = nn.Identity()
        self.relu = nn.ReLU()
        self.reshape = ReshapeOutput()
        
        self.header = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(768,self.cls_nums),
            nn.Identity()
        )
    def forward(self,inputs):
        A = self.fe(inputs)
        # A = self.id(A)
        A = self.relu(A)
        A = self.reshape(A)
        
        O = self.header(A)
        return O
     
class Deit_Tiny(nn.Module) :
    def __init__(self,cls_nums):
        super (Deit_Tiny,self).__init__()
        self.cls_nums = cls_nums
        self.fe = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='')
        # self.id = nn.Identity()
        self.relu = nn.ReLU()
        self.reshape = ReshapeOutput()
        
        self.header = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(192,self.cls_nums),
            nn.Identity()
        )
    def forward(self,inputs):
        A = self.fe(inputs)
        # A = self.id(A)
        A = self.relu(A)
        A = self.reshape(A)
        
        O = self.header(A)
        return O

class Deit_Base(nn.Module) :
    def __init__(self,cls_nums):
        super (Deit_Base,self).__init__()
        self.cls_nums = cls_nums
        self.fe = timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=0, global_pool='')
        # self.id = nn.Identity()
        self.relu = nn.ReLU()
        self.reshape = ReshapeOutput()
        
        self.header = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(768,self.cls_nums),
            nn.Identity()
        )
    def forward(self,inputs):
        A = self.fe(inputs)
        # A = self.id(A)
        A = self.relu(A)
        A = self.reshape(A)
        
        O = self.header(A)
        return O

class NormLinear(nn.Module):
    def __init__(self,in_dims,cls_nums):
        super (NormLinear,self).__init__()
        self.bn = nn.BatchNorm1d(in_dims)
        self.linear = nn.Linear(in_dims,cls_nums)
    
    def forward(self,inputs):
        return self.linear(self.bn(inputs))
        
class Levit(nn.Module) :
    def __init__(self,cls_nums):
        super (Levit,self).__init__()
        self.cls_nums = cls_nums
        self.fe = timm.create_model('levit_256', pretrained=True, num_classes=0, global_pool='')
        # self.id = nn.Identity()
        self.relu = nn.ReLU()
        self.reshape = ReshapeOutput()
        
        self.header = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            NormLinear(512,self.cls_nums),
            nn.Identity()
        )
    def forward(self,inputs):
        A = self.fe(inputs)
        # A = self.id(A)
        A = self.relu(A)
        A = self.reshape(A)
        
        O = self.header(A)
        return O

class GCVIT(nn.Module) :
    def __init__(self,cls_nums):
        super (GCVIT,self).__init__()
        self.cls_nums = cls_nums
        self.fe = timm.create_model('gcvit_base', pretrained=True, num_classes=0, global_pool='')
        self.id = nn.Identity()
        self.relu = nn.ReLU()
        self.header = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024,self.cls_nums),
            nn.Identity()
        )
    def forward(self,inputs):
        A = self.fe(inputs)
        A = self.id(A)
        A = self.relu(A)
        O = self.header(A)
        return O

class PREDICTOR(nn.Module):
    def __init__(self,concepts_nums,cls_nums):
        super (PREDICTOR,self).__init__()
        self.concepts_nums = concepts_nums
        self.cls_nums = cls_nums
        self.weights = nn.Parameter(torch.randn((self.concepts_nums,self.cls_nums)))
        # self.relu = nn.ReLU()
        self.linear = nn.Linear(self.cls_nums,self.cls_nums)
        self.id = nn.Identity()
    def forward(self,inputs):
        return self.id(self.linear(inputs @ self.weights))
    
class PREDICTOR_NL(nn.Module):
    def __init__(self,concepts_nums,cls_nums):
        super (PREDICTOR_NL,self).__init__()
        self.concepts_nums = concepts_nums
        self.cls_nums = cls_nums
        self.weights = nn.Parameter(torch.randn((self.concepts_nums,self.cls_nums)))
        # self.relu = nn.ReLU()
        self.A = nn.Parameter(torch.randn((self.cls_nums)))
        self.B = nn.Parameter(torch.randn((self.cls_nums)))
    def forward(self,inputs):
        return self.A * (inputs @ self.weights) + self.B