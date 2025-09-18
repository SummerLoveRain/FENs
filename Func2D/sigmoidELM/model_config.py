from cmath import inf
import time
import numpy as np
import torch
from base_config import BaseConfig

class PINNConfig(BaseConfig):
    def __init__(self, param_dict, train_dict, model):
        super().__init__(**param_dict)
        self.model = model

        # 加载训练参数
        x, y, x_valid, y_valid, self.funU, self.N_basis = self.unzip_train_dict(
            train_dict=train_dict)

        # 区域内点
        self.x = self.data_loader(x)
        self.y = self.data_loader(y)
        # 验证
        self.x_valid = self.data_loader(x_valid)
        self.y_valid = self.data_loader(y_valid)
    
        self.u_true = self.reload(self.funU(self.x, self.y), requires_grad=False)
        self.u_valid = self.reload(self.funU(self.x_valid, self.y_valid), requires_grad=False)
        
        self.W = None
        self.params = list(self.model.parameters())


    def unzip_train_dict(self, train_dict):
        train_data = (
            train_dict['x'],
            train_dict['y'],
            train_dict['x_valid'],
            train_dict['y_valid'],
            train_dict['funU'],
            train_dict['N_basis'],
        )
        return train_data

    def net_model(self, x, y):
        X = torch.cat((x, y), dim=1)
        X = self.coor_shift(X, self.lb, self.ub)
        us = self.model.forward(X)
        # us = torch.tanh(us)
        return us

    def forward(self, x, y):
        us = self.net_model(x, y)
        u = torch.matmul(us, self.W)
        return u

    def lstsq(self, A, b):        
        W = torch.linalg.lstsq(A, b)[0]
        return W
    
    # 训练一次
    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        u_true = self.u_true
        # 初始化loss为0
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        self.loss.requires_grad_()

        # 区域点
        x = self.x
        y = self.y
        us = self.net_model(x, y)
        # with torch.no_grad():
        
        A = us
        b = u_true
        W = self.lstsq(A=A, b=u_true)
        self.W = self.reload(W, requires_grad=False)

        # 打印耗时
        elapsed = time.time() - self.start_time
        log_str = 'Elapsed Time: %.4fs' % (elapsed)
        self.log(log_str)
        
        # A=us
        # rank = np.max(self.detach(torch.linalg.matrix_rank(us)))
        # self.log('rank ' + str(rank))


        u = torch.matmul(us, self.W)
        u_infty = np.max(self.detach(torch.linalg.norm(u_true-u, ord=inf)))
        u_rel_l2 = np.max(self.detach(torch.linalg.norm(u_true-u)/torch.linalg.norm(u_true)))           
        log_str = 'u_infty ' + str(u_infty) + ' u_rel_l2 ' + str(u_rel_l2)
        self.log(log_str)

        # 验证
        u_true = self.u_valid
        u = self.forward(self.x_valid, self.y_valid)
    
        u_infty = np.max(self.detach(torch.linalg.norm(u_true-u, ord=inf)))
        u_rel_l2 = np.max(self.detach(torch.linalg.norm(u_true-u)/torch.linalg.norm(u_true)))           
        log_str = 'u_infty ' + str(u_infty) + ' u_rel_l2 ' + str(u_rel_l2)
        self.log(log_str)

        Aw = torch.matmul(A, W)
        loss_infty = np.max(self.detach(torch.linalg.norm(b - Aw, ord=inf)))
        loss_rel_l2 = np.max(self.detach(torch.linalg.norm(b - Aw)/torch.linalg.norm(b)))
        log_str = 'loss_infty ' + str(loss_infty) + ' loss_rel_l2 ' + str(loss_rel_l2)
        self.log(log_str)
    
        # 运算次数加1
        self.nIter = self.nIter + 1
        loss_remainder = 1

        # 打印耗时
        elapsed = time.time() - self.start_time
        log_str = 'Time: %.4fs Per %d Iterators' % (elapsed, loss_remainder)
        self.log(log_str)
        self.start_time = time.time()

        # PINNConfig.save(net=self,
        #                 path=self.root_path + '/' + self.path,
        #                 name=self.model_name)

        return self.loss
