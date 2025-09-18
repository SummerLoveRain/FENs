import torch.nn as nn        
import torch
from torch import Tensor
import numpy as np
import math
import warnings
from torch.nn.parameter import Parameter

class MODEL_NAME:
    '''
    常量类，定义神经网络名称
    '''
    MLP = 'MLP'
    ResNet = 'ResNet'
    MultiMLPs = 'MultiMLPs'
    ResNetv2 = 'ResNetv2'
    ResNetv3 = 'ResNetv3'
    ResNetv4 = 'ResNetv4'
    MultiModels = 'MultiModels'
    FourierNet = 'FourierNet'
    tanhELM = 'tanhELM'
    sigmoidELM = 'sigmoidELM'
    swishELM = 'swishELM'
    cosELM = 'cosELM'
    sinELM = 'sinELM'
    csELM = 'csELM'

def get_model(model_dict, model_name):
    '''
    获取神经网络模型
    '''
    model = None
    if model_name == MODEL_NAME.MLP:
        model = MLP(**model_dict)
    elif model_name == MODEL_NAME.ResNet:
        model = ResNet(**model_dict)
    elif model_name == MODEL_NAME.MultiMLPs:
        model = MultiMLPs(**model_dict)
    elif model_name == MODEL_NAME.ResNetv2:
        model = ResNetv2(**model_dict)
    elif model_name == MODEL_NAME.ResNetv3:
        model = ResNetv3(**model_dict)
    elif model_name == MODEL_NAME.ResNetv4:
        model = ResNetv4(**model_dict)
    elif model_name == MODEL_NAME.MultiModels:
        model = MultiModels(**model_dict)
    elif model_name == MODEL_NAME.FourierNet:
        model = FourierNet(**model_dict)
    elif model_name == MODEL_NAME.tanhELM:
        model = tanhELM(**model_dict)
    elif model_name == MODEL_NAME.sigmoidELM:
        model = sigmoidELM(**model_dict)
    elif model_name == MODEL_NAME.swishELM:
        model = swishELM(**model_dict)
    elif model_name == MODEL_NAME.cosELM:
        model = cosELM(**model_dict)
    elif model_name == MODEL_NAME.sinELM:
        model = sinELM(**model_dict)
    elif model_name == MODEL_NAME.csELM:
        model = csELM(**model_dict)
    else:
        ValueError("The model_name is not correct!")
    return model

class INIT_TYPE:
    '''
    常量类，定义初始化名称
    '''
    Normal = 'normal'
    Uniform = 'uniform'
    Xavier_Uniform = 'Xavier_uniform'
    Xavier_Normal = 'Xavier_normal'
    Kaiming_Uniform = 'Kaiming_uniform'
    Kaiming_Normal = 'Kaiming_normal'
    Qihong_Uniform = 'Qihong_uniform'
    Qihong_Normal = 'Qihong_normal'

class Base_Model(nn.Module):   
    def __init__(self, init_type, init_params):
        super().__init__()
        self.init_type = init_type
        self.init_params = init_params
    # 模型初始化
    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            if INIT_TYPE.Normal == self.init_type:
                self.normal_(m.weight, m.bias, **self.init_params)
            elif INIT_TYPE.Uniform == self.init_type:
                self.uniform_(m.weight, m.bias, **self.init_params)            
            elif INIT_TYPE.Xavier_Normal == self.init_type:
                self.xavier_normal_(m.weight, m.bias, **self.init_params)
            elif INIT_TYPE.Xavier_Uniform == self.init_type:
                self.xavier_uniform_(m.weight, m.bias, **self.init_params)
            elif INIT_TYPE.Kaiming_Normal == self.init_type:
                self.kaiming_normal_(m.weight, m.bias, **self.init_params)
            elif INIT_TYPE.Kaiming_Uniform == self.init_type:
                self.kaiming_uniform_(m.weight, m.bias, **self.init_params) 
            elif INIT_TYPE.Qihong_Normal == self.init_type:
                self.Qihong_normal_(m.weight, m.bias, **self.init_params)
            elif INIT_TYPE.Qihong_Uniform == self.init_type:
                self.Qihong_uniform_(m.weight, m.bias, **self.init_params)
            else:
                nn.init.xavier_normal_(m.weight)
                # nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def init_tensor(self, W, b=None):
        if b is None:
            b = W            
        if INIT_TYPE.Normal == self.init_type:
            self.normal_(W, b, **self.init_params)
        elif INIT_TYPE.Uniform == self.init_type:
            self.uniform_(W, b, **self.init_params)            
        elif INIT_TYPE.Xavier_Normal == self.init_type:
            self.xavier_normal_(W, b, **self.init_params)
        elif INIT_TYPE.Xavier_Uniform == self.init_type:
            self.xavier_uniform_(W, b, **self.init_params)
        elif INIT_TYPE.Kaiming_Normal == self.init_type:
            self.kaiming_normal_(W, b, **self.init_params)
        elif INIT_TYPE.Kaiming_Uniform == self.init_type:
            self.kaiming_uniform_(W, b, **self.init_params) 
        elif INIT_TYPE.Qihong_Normal == self.init_type:
            self.Qihong_normal_(W, b, **self.init_params)
        elif INIT_TYPE.Qihong_Uniform == self.init_type:
            self.Qihong_uniform_(W, b, **self.init_params)
        else:
            # nn.init.xavier_normal_(tensor)
            # nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(W, 0)
            nn.init.constant_(b, 0)


    def uniform_(self, weight: Tensor, bias: Tensor, var: float=1, gain: float=1):
        r = np.sqrt(3*var)*gain
        nn.init.uniform_(weight, -r, r)
        nn.init.uniform_(bias, -r, r)

    def normal_(self, weight: Tensor, bias: Tensor, var: float=1, gain: float=1):
        r = np.sqrt(var)*gain
        nn.init._no_grad_normal_(weight, 0, r)
        nn.init._no_grad_normal_(bias, 0, r)

    def Qihong_uniform_(self, weight: Tensor, bias: Tensor, var_x: float=0.5, gain: float=1):
        in_dim = nn.init._calculate_correct_fan(weight, mode='fan_in')
        var = (2 - 2*(1-var_x)**2)**2/(np.pi**2*var_x*(1-var_x)**4*in_dim)
        var_b = var
        var_W = var
        r_W = np.sqrt(3*var_W)*gain
        r_b = np.sqrt(3*var_b)*gain
        nn.init.uniform_(weight, -r_W, r_W)
        nn.init.uniform_(bias, -r_b, r_b)

    def Qihong_normal_(self, weight: Tensor, bias: Tensor, var_x: float=0.5, gain: float=1):
        in_dim = nn.init._calculate_correct_fan(weight, mode='fan_in')
        var = (2 - 2*(1-var_x)**2)**2/(np.pi**2*var_x*(1-var_x)**4*in_dim)
        var_b = var
        var_W = var
        r_W = np.sqrt(var_W)*gain
        r_b = np.sqrt(var_b)*gain
        nn.init._no_grad_normal_(weight, 0, r_W)
        nn.init._no_grad_normal_(bias, 0, r_b)

    def xavier_normal_(self, weight: Tensor, bias: Tensor, gain: float = 1.):
        dimensions = weight.dim()
        if dimensions < 2:
            fan_in = 1
        else:
            fan_in = weight.shape[1]
        fan_out = weight.shape[0]
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        nn.init._no_grad_normal_(weight, 0., std)
        nn.init._no_grad_normal_(bias, 0., std)

        
    def xavier_uniform_(self, weight: Tensor, bias: Tensor, gain: float = 1.):
        dimensions = weight.dim()
        if dimensions < 2:
            fan_in = 1
        else:
            fan_in = weight.shape[1]
        fan_out = weight.shape[0]
        r = gain * math.sqrt(6.0 / float(fan_in + fan_out))
        nn.init._no_grad_uniform_(weight, -r, r)
        nn.init._no_grad_uniform_(bias, -r, r)
    
    def kaiming_uniform_(self, weight: Tensor, bias: Tensor, gain: float = 1.):
        fan_in = nn.init._calculate_correct_fan(weight, mode='fan_in')
        r = gain * math.sqrt(3.0 / float(fan_in))
        nn.init._no_grad_uniform_(weight, -r, r)
        nn.init._no_grad_uniform_(bias, -r, r)

    def kaiming_normal_(self, weight: Tensor, bias: Tensor, gain: float = 1.):
        fan_in = nn.init._calculate_correct_fan(weight, mode='fan_in')
        r = gain * math.sqrt(1.0 / float(fan_in))
        nn.init._no_grad_normal_(weight, 0., r)
        nn.init._no_grad_normal_(bias, 0., r)

    
class MLP(Base_Model):
    def __init__(self, layers, act_func=nn.Tanh(), init_type=INIT_TYPE.Xavier_Normal, init_params={}):
        super().__init__(init_type, init_params)
        self.layers = layers
        self.act_func = act_func
        self.linear_list = []
        for i in range(len(self.layers)-2):
            linear = nn.Linear(self.layers[i],self.layers[i+1])
            self.weight_init(linear)
            self.linear_list.append(linear)
        self.linear_list = nn.ModuleList(self.linear_list)
        linear = nn.Linear(self.layers[-2],self.layers[-1])
        self.weight_init(linear)
        self.fc = linear

    def forward(self, x):
        for i in range(len(self.linear_list)):
            linear = self.linear_list[i]
            x = self.act_func(linear(x))
        linear = self.fc
        y = linear(x)
        return y


class ResNet(Base_Model):
    # in_num, 输入神经元个数
    # out_num, 输出神经元个数
    # block_layers, 残差中的全连接层结构
    # block_num, 残差模块的个数
    def __init__(self, in_num, out_num, block_layers, block_num, act_func=nn.Tanh(), init_type=INIT_TYPE.Xavier_Normal, init_params={}):
        super().__init__(init_type, init_params)

        self.block_layers = block_layers
        self.block_num = block_num

        self.in_linear = nn.Linear(in_num, self.block_layers[0])
        self.out_linear = nn.Linear(self.block_layers[-1], out_num)

        self.act_func = act_func

        self.jump_list = []
        self.mlps = []
        for _ in range(self.block_num):
            if self.block_layers[0] == self.block_layers[-1]:
                jump_linear = nn.Identity()
            else:
                jump_linear = nn.Linear(self.block_layers[0], self.block_layers[-1])
            self.weight_init(jump_linear)
            self.jump_list.append(jump_linear)
            mlp = MLP(block_layers, self.act_func)
            self.mlps.append(mlp)
        self.jump_list = nn.ModuleList(self.jump_list)
        self.mlps = nn.ModuleList(self.mlps)


    def forward(self, x):
        x = self.in_linear(x)
        for i in range(self.block_num):
            mlp = self.mlps[i]
            jump_linear = self.jump_list[i]
            x = mlp(x) + jump_linear(x)
            x = self.act_func(x)
        y = self.out_linear(x)
        return y


class MultiMLPs(Base_Model):
    def __init__(self, mlp_layers, act_func=nn.Tanh(), init_type=INIT_TYPE.Xavier_Normal, init_params={}):
        super().__init__(init_type, init_params)
        self.num_mlp = len(mlp_layers)
        self.mlp_layers = mlp_layers
        self.act_func = act_func
        self.mlps = []
        for i in range(self.num_mlp):
            mlp = MLP(self.mlp_layers[i], self.act_func)
            self.mlps.append(mlp)
        self.mlps = nn.ModuleList(self.mlps)
    
    def mlp(self, index, x):
        return self.mlps[index](x)

    def forward(self, x):
        out = []
        for i in range(self.num_mlp):
            out.append(self.mlps[i](x))
        out = torch.cat((out), dim=1)
        return out


class ResNetv2(Base_Model):
    # in_num, 输入神经元个数
    # out_num, 输出神经元个数
    # block_layers, 残差中的全连接层结构
    # block_num, 残差模块的个数
    def __init__(self, in_num, out_num, block_layers, block_num, act_func=nn.Tanh(), init_type=INIT_TYPE.Xavier_Normal, init_params={}):
        super().__init__(init_type, init_params)

        self.block_layers = block_layers
        self.block_num = block_num

        self.in_linear = nn.Linear(in_num, self.block_layers[0])
        self.out_linear = nn.Linear(self.block_layers[-1], out_num)

        self.act_func = act_func

        self.jump_list = []
        self.mlps = []
        for _ in range(self.block_num):
            if self.block_layers[0] == self.block_layers[-1]:
                jump_linear = nn.Identity()
            else:
                jump_linear = nn.Linear(self.block_layers[0], self.block_layers[-1])
            self.weight_init(jump_linear)
            self.jump_list.append(jump_linear)
            mlp = MLP(block_layers, self.act_func)
            self.mlps.append(mlp)
        self.mlps = nn.ModuleList(self.mlps)


    def forward(self, x):
        x = self.in_linear(x)
        tmp_x = x
        for i in range(self.block_num):
            mlp = self.mlps[i]
            jump_linear = self.jump_list[i]
            x = mlp(x) + jump_linear(x)
            if i != 0:
                x = x + tmp_x
            x = self.act_func(x)
        y = self.out_linear(x)
        return y


class ResNetv3(Base_Model):
    # in_num, 输入神经元个数
    # out_num, 输出神经元个数
    # block_layers, 残差中的全连接层结构
    # block_num, 残差模块的个数
    def __init__(self, layers, act_func=nn.Tanh(), init_type=INIT_TYPE.Xavier_Normal, init_params={}):
        super().__init__(init_type, init_params)

        self.layers = layers
        self.in_num = layers[0]
        self.out_num = layers[-1]

        self.out_linear = nn.Linear(self.layers[-2], self.out_num)

        self.act_func = act_func

        self.jump_list = []
        self.mlps = []
        for i in range(len(self.layers)-2):
            # 设置跳连
            if self.layers[i] == self.layers[i+1]:
                jump_linear = nn.Identity()
            else:
                jump_linear = nn.Linear(self.layers[i], self.layers[i+1])
            self.weight_init(jump_linear)
            self.jump_list.append(jump_linear)
            linear = nn.Linear(self.layers[i], self.layers[i+1])
            self.mlps.append(linear)
        self.jump_list = nn.ModuleList(self.jump_list)
        self.mlps = nn.ModuleList(self.mlps)


    def forward(self, x):
        for i in range(len(self.layers)-2):
            mlp = self.mlps[i]
            jump_linear = self.jump_list[i]
            x = self.act_func(mlp(x)) + jump_linear(x)
            x = self.act_func(x)
        y = self.out_linear(x)
        return y
            

class ResNetv4(Base_Model):
    # in_num, 输入神经元个数
    # out_num, 输出神经元个数
    # block_layers, 残差中的全连接层结构
    # block_num, 残差模块的个数
    def __init__(self, in_num, out_num, block_layers, act_func=nn.Tanh(), init_type=INIT_TYPE.Xavier_Normal, init_params={}):
        super().__init__(init_type, init_params)

        self.in_num = in_num
        self.out_num = out_num
        self.block_layers = block_layers
        self.block_num = len(block_layers)
        
        self.in_linear = nn.Linear(self.in_num, self.block_layers[0][0])
        self.out_linear = nn.Linear(self.block_layers[-1][-1], self.out_num)

        self.act_func = act_func

        self.act_func = act_func

        self.jump_list = []
        self.mlps = []
        for block_layer in self.block_layers:
            if block_layer[0] == block_layer[-1]:
                jump_linear = nn.Identity()
            else:
                jump_linear = nn.Linear(block_layer[0], block_layer[-1])
            self.weight_init(jump_linear)
            self.jump_list.append(jump_linear)
            mlp = MLP(block_layer, self.act_func)
            self.mlps.append(mlp)
        self.jump_list = nn.ModuleList(self.jump_list)
        self.mlps = nn.ModuleList(self.mlps)


    def forward(self, x):
        x = self.in_linear(x)
        x = self.act_func(x)
        for i in range(self.block_num):
            mlp = self.mlps[i]
            jump_linear = self.jump_list[i]
            # x = mlp(x) + jump_linear(x)
            x = self.act_func(mlp(x)) + jump_linear(x)
            x = self.act_func(x)
        y = self.out_linear(x)
        return y

class MultiModels(nn.Module):
    def __init__(self, model_dicts, model_names):
        super().__init__()
        self.model_dicts = model_dicts
        self.model_names = model_names
        self.models = []
        for model_dict, model_name in zip(self.model_dicts, self.model_names):
            model = get_model(model_dict=model_dict, model_name=model_name)
            self.models.append(model)
        self.models = nn.ModuleList(self.models)
    
    def forward(self, x):
        out = []
        for model in self.models:
            out.append(model(x))
        out = torch.cat((out), dim=1)
        return out
    
class FourierNet(Base_Model):
    def __init__(self, layers, scale=1, init_type=INIT_TYPE.Xavier_Normal, init_params={}):
        super().__init__(init_type, init_params)
        self.layers = layers
        self.in_num = layers[0]
        self.out_num = layers[1]        
        self.N_basis_F = self.out_num//2
        
        self.scale = Parameter(torch.ones((1, 1))*scale)
        self.omega_n = Parameter(torch.empty((self.N_basis_F, self.in_num)))
        self.init_tensor(self.omega_n)
        
    def diff_x(self, x, idx, shift_coeff=1):
        trans_omega_n = torch.transpose(self.omega_n, dim0=0, dim1=1)*self.scale
        sin_us = torch.sin(2*torch.pi*torch.matmul(x, trans_omega_n))
        cos_us = torch.cos(2*torch.pi*torch.matmul(x, trans_omega_n))
        
        trans_omega_n_idx = trans_omega_n[idx:idx+1, :] * shift_coeff
        diff_sin = 2*torch.pi*trans_omega_n_idx * cos_us
        diff_cos = -2*torch.pi*trans_omega_n_idx * sin_us
        y = torch.cat((diff_sin, diff_cos), dim=1)
        return y
    

    def diff_xx(self, x, idx, shift_coeff=1):
        trans_omega_n = torch.transpose(self.omega_n, dim0=0, dim1=1)*self.scale
        sin_us = torch.sin(2*torch.pi*torch.matmul(x, trans_omega_n))
        cos_us = torch.cos(2*torch.pi*torch.matmul(x, trans_omega_n))
        
        trans_omega_n_idx = trans_omega_n[idx:idx+1, :] * shift_coeff
        diff_sin = -(2*torch.pi*trans_omega_n_idx)**2 * sin_us
        diff_cos = -(2*torch.pi*trans_omega_n_idx)**2 * cos_us
        y = torch.cat((diff_sin, diff_cos), dim=1)
        return y
        

    def forward(self, x):
        trans_omega_n = torch.transpose(self.omega_n, dim0=0, dim1=1)*self.scale
        sin_us = torch.sin(2*torch.pi*torch.matmul(x, trans_omega_n))
        cos_us = torch.cos(2*torch.pi*torch.matmul(x, trans_omega_n))
        y = torch.cat((sin_us, cos_us), dim=1)
        return y

    
class tanhELM(Base_Model):
    def __init__(self, layers, scale=1, init_type=INIT_TYPE.Xavier_Normal, init_params={}):
        super().__init__(init_type, init_params)
        self.layers = layers
        self.in_num = layers[0]
        self.out_num = layers[1]        
        self.N_basis_F = self.out_num
        self.act_func = nn.Tanh()
        
        self.scale = Parameter(torch.ones((1, 1))*scale)
        self.omega = Parameter(torch.empty((self.N_basis_F, self.in_num)))
        self.b = Parameter(torch.empty(1, self.N_basis_F))
        self.init_tensor(self.omega, self.b)
    
    def diff_x(self, x, idx, shift_coeff=1):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        us = self.act_func(torch.matmul(x, trans_omega) + self.b*self.scale)
        
        trans_omega_idx = trans_omega[idx:idx+1, :] * shift_coeff
        diff_us = trans_omega_idx * (1-us**2)
        return diff_us
    
    def diff_xx(self, x, idx, shift_coeff=1):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        us = self.act_func(torch.matmul(x, trans_omega) + self.b*self.scale)
        
        trans_omega_idx = trans_omega[idx:idx+1, :] * shift_coeff
        diff_us = 2*trans_omega_idx**2 * (us**3-us)
        return diff_us
        

    def forward(self, x):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        us = self.act_func(torch.matmul(x, trans_omega) + self.b*self.scale)
        return us
    

class sigmoidELM(Base_Model):
    def __init__(self, layers, scale=1, init_type=INIT_TYPE.Xavier_Normal, init_params={}):
        super().__init__(init_type, init_params)
        self.layers = layers
        self.in_num = layers[0]
        self.out_num = layers[1]        
        self.N_basis_F = self.out_num
        self.act_func = nn.Sigmoid()
        
        self.scale = Parameter(torch.ones((1, 1))*scale)
        self.omega = Parameter(torch.empty((self.N_basis_F, self.in_num)))
        self.b = Parameter(torch.empty(1, self.N_basis_F))
        self.init_tensor(self.omega, self.b)
        
    def diff_x(self, x, idx, shift_coeff=1):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        us = self.act_func(torch.matmul(x, trans_omega) + self.b*self.scale)
        
        trans_omega_idx = trans_omega[idx:idx+1, :] * shift_coeff
        diff_us = trans_omega_idx * (us-us**2)
        return diff_us
    

    def diff_xx(self, x, idx, shift_coeff=1):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        us = self.act_func(torch.matmul(x, trans_omega) + self.b*self.scale)
        
        trans_omega_idx = trans_omega[idx:idx+1, :] * shift_coeff
        diff_us = trans_omega_idx**2 * (us-3*us**2+2*us**3)
        return diff_us
        

    def forward(self, x):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        y = self.act_func(torch.matmul(x, trans_omega) + self.b*self.scale)
        return y
    
class swishELM(Base_Model):
    def __init__(self, layers, scale=1, init_type=INIT_TYPE.Xavier_Normal, init_params={}):
        super().__init__(init_type, init_params)
        self.layers = layers
        self.in_num = layers[0]
        self.out_num = layers[1]        
        self.N_basis_F = self.out_num
        self.act_func = nn.Sigmoid()
        
        self.scale = Parameter(torch.ones((1, 1))*scale)
        self.omega = Parameter(torch.empty((self.N_basis_F, self.in_num)))
        self.b = Parameter(torch.empty(1, self.N_basis_F))
        self.init_tensor(self.omega, self.b)
        
    def diff_x(self, x, idx, shift_coeff=1):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        y = torch.matmul(x, trans_omega) + self.b*self.scale
        us = self.act_func(y)
        
        trans_omega_idx = trans_omega[idx:idx+1, :] * shift_coeff
        diff_us = trans_omega_idx * (us+y*us-y*us**2)
        return diff_us
    

    def diff_xx(self, x, idx, shift_coeff=1):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        y = torch.matmul(x, trans_omega) + self.b*self.scale
        us = self.act_func(y)
        
        trans_omega_idx = trans_omega[idx:idx+1, :] * shift_coeff
        diff_us = trans_omega_idx**2 * (2*us-2*us**2+y*us-3*y*us**2+2*y*us**3)
        return diff_us
        

    def forward(self, x):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        y = torch.matmul(x, trans_omega) + self.b*self.scale
        y = y*self.act_func(y)
        return y
    

class cosELM(Base_Model):
    def __init__(self, layers, scale=1, init_type=INIT_TYPE.Xavier_Normal, init_params={}):
        super().__init__(init_type, init_params)
        self.layers = layers
        self.in_num = layers[0]
        self.out_num = layers[1]        
        self.N_basis_F = self.out_num
        self.act_func = torch.cos
        
        self.scale = Parameter(torch.ones((1, 1))*scale)
        self.omega = Parameter(torch.empty((self.N_basis_F, self.in_num)))
        self.b = Parameter(torch.empty(1, self.N_basis_F))
        self.init_tensor(self.omega, self.b)
        
    def diff_x(self, x, idx, shift_coeff=1):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        us = torch.sin(torch.matmul(x, trans_omega) + self.b*self.scale)
        
        trans_omega_idx = trans_omega[idx:idx+1, :] * shift_coeff
        diff_us = trans_omega_idx * (-us)
        return diff_us
    

    def diff_xx(self, x, idx, shift_coeff=1):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        us = self.act_func(torch.matmul(x, trans_omega) + self.b*self.scale)
        
        trans_omega_idx = trans_omega[idx:idx+1, :] * shift_coeff
        diff_us = -trans_omega_idx**2 * us
        return diff_us
        

    def forward(self, x):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        y = self.act_func(torch.matmul(x, trans_omega) + self.b*self.scale)
        return y
    
    
class sinELM(Base_Model):
    def __init__(self, layers, scale=1, init_type=INIT_TYPE.Xavier_Normal, init_params={}):
        super().__init__(init_type, init_params)
        self.layers = layers
        self.in_num = layers[0]
        self.out_num = layers[1]        
        self.N_basis_F = self.out_num
        self.act_func = torch.sin
        
        self.scale = Parameter(torch.ones((1, 1))*scale)
        self.omega = Parameter(torch.empty((self.N_basis_F, self.in_num)))
        self.b = Parameter(torch.empty(1, self.N_basis_F))
        self.init_tensor(self.omega, self.b)
        
    def diff_x(self, x, idx, shift_coeff=1):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        us = torch.cos(torch.matmul(x, trans_omega) + self.b*self.scale)
        
        trans_omega_idx = trans_omega[idx:idx+1, :] * shift_coeff
        diff_us = trans_omega_idx * us
        return diff_us
    

    def diff_xx(self, x, idx, shift_coeff=1):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        us = self.act_func(torch.matmul(x, trans_omega) + self.b*self.scale)
        
        trans_omega_idx = trans_omega[idx:idx+1, :] * shift_coeff
        diff_us = -trans_omega_idx**2 * us
        return diff_us
        

    def forward(self, x):
        trans_omega = torch.transpose(self.omega, dim0=0, dim1=1)*self.scale
        y = self.act_func(torch.matmul(x, trans_omega) + self.b*self.scale)
        return y
    

class csELM(Base_Model):
    def __init__(self, layers, scale=1, init_type=INIT_TYPE.Xavier_Normal, init_params={}):
        super().__init__(init_type, init_params)
        self.layers = layers
        self.in_num = layers[0]
        self.out_num = layers[1]        
        self.N_basis_F = self.out_num//2
        
        self.scale = Parameter(torch.ones((1, 1))*scale)
        self.omega1 = Parameter(torch.empty((self.N_basis_F, self.in_num)))
        self.b1 = Parameter(torch.empty(1, self.N_basis_F))
        self.omega2 = Parameter(torch.empty((self.N_basis_F, self.in_num)))
        self.b2 = Parameter(torch.empty(1, self.N_basis_F))
        self.init_tensor(self.omega1, self.b1)
        self.init_tensor(self.omega2, self.b2)
        
    def diff_x(self, x, idx, shift_coeff=1):
        trans_omega1 = torch.transpose(self.omega1, dim0=0, dim1=1)*self.scale
        trans_omega2 = torch.transpose(self.omega2, dim0=0, dim1=1)*self.scale
        sin_us = torch.sin(torch.matmul(x, trans_omega1) + self.b1*self.scale)
        cos_us = torch.cos(torch.matmul(x, trans_omega2) + self.b2*self.scale)
        
        trans_omega1_idx = trans_omega1[idx:idx+1, :] * shift_coeff
        trans_omega2_idx = trans_omega2[idx:idx+1, :] * shift_coeff
        diff_cos = -trans_omega1_idx * sin_us
        diff_sin = trans_omega2_idx * cos_us
        y = torch.cat((diff_cos, diff_sin), dim=1)
        return y
    

    def diff_xx(self, x, idx, shift_coeff=1):
        trans_omega1 = torch.transpose(self.omega1, dim0=0, dim1=1)*self.scale
        trans_omega2 = torch.transpose(self.omega2, dim0=0, dim1=1)*self.scale
        cos_us = torch.cos(torch.matmul(x, trans_omega1) + self.b1*self.scale)
        sin_us = torch.sin(torch.matmul(x, trans_omega2) + self.b2*self.scale)
        
        trans_omega1_idx = trans_omega1[idx:idx+1, :] * shift_coeff
        trans_omega2_idx = trans_omega2[idx:idx+1, :] * shift_coeff
        diff_cos = -trans_omega1_idx**2 * cos_us
        diff_sin = -trans_omega2_idx**2 * sin_us
        y = torch.cat((diff_cos, diff_sin), dim=1)
        return y
        

    def forward(self, x):
        trans_omega1 = torch.transpose(self.omega1, dim0=0, dim1=1)*self.scale
        trans_omega2 = torch.transpose(self.omega2, dim0=0, dim1=1)*self.scale
        cos_us = torch.cos(torch.matmul(x, trans_omega1) + self.b1*self.scale)
        sin_us = torch.sin(torch.matmul(x, trans_omega2) + self.b2*self.scale)
        y = torch.cat((cos_us, sin_us), dim=1)
        return y