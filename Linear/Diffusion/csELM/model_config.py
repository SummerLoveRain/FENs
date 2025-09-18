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
        self.nu, x, t, xb, tb, x0, t0, x_valid, t_valid, self.funF, self.funU, self.N_basis = self.unzip_train_dict(
            train_dict=train_dict)

        # 区域内点
        self.x = self.data_loader(x)
        self.t = self.data_loader(t)
        self.xb = self.data_loader(xb)
        self.tb = self.data_loader(tb)
        self.x0 = self.data_loader(x0)
        self.t0 = self.data_loader(t0)

        self.u_true = self.data_loader(self.funU(x, t, type='np'), requires_grad=False)
        self.u_valid = self.data_loader(self.funU(x_valid, t_valid, type='np'), requires_grad=False)
        self.x_valid = self.data_loader(x_valid, requires_grad=False)
        self.t_valid = self.data_loader(t_valid, requires_grad=False)

        self.W = self.xavier_init([self.N_basis, 1])
        self.params = [self.W]
        # self.params = list(self.model.parameters()) + [self.W]

        A = np.zeros([x.shape[0], self.N_basis])
        self.A = self.data_loader(A, requires_grad=False)

    def unzip_train_dict(self, train_dict):
        train_data = (
            train_dict['nu'],
            train_dict['x'],
            train_dict['t'],
            train_dict['xb'],
            train_dict['tb'],
            train_dict['x0'],
            train_dict['t0'],
            train_dict['x_valid'],
            train_dict['t_valid'],
            train_dict['funF'],
            train_dict['funU'],
            train_dict['N_basis'],
        )
        return train_data

    def net_model(self, x, t):
        X = torch.cat((x, t), dim=1)
        # X = self.coor_shift(X, self.lb, self.ub)
        us = self.model.forward(X)
        # us = torch.tanh(us)
        # us = self.model.act_func(us)
        return us

    def forward(self, x, t):
        us = self.net_model(x, t)
        u = torch.matmul(us, self.W)
        return u
    
    def lstsq(self, A, b):        
        W = torch.linalg.lstsq(A, b)[0]
        return W
    
    def diff_x(self, x, t, idx):    
        X = torch.cat((x, t), dim=1)
        # X = self.coor_shift(X, self.lb, self.ub)
        us = self.model.diff_x(X, idx)
        # us = torch.tanh(us)
        # us = self.model.act_func(us)
        return us
    
    def diff_xx(self, x, t, idx):
        X = torch.cat((x, t), dim=1)
        # X = self.coor_shift(X, self.lb, self.ub)
        us = self.model.diff_xx(X, idx)
        # us = torch.tanh(us)
        # us = self.model.act_func(us)
        return us
    
    
    def Lus(self, us, x, t):
        us_xx = self.diff_xx(x, t, idx=0)
        us_t = self.diff_x(x, t, idx=1)

        Lus = us_t - self.nu*us_xx
        return Lus

    def Lu(self, u, x, t):
        u_t = self.compute_grad(u, t)
        u_x = self.compute_grad(u, x)
        u_xx = self.compute_grad(u_x, x)
        Lu = u_t - self.nu*u_xx
        return Lu
    
    
    # 训练一次
    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        # 初始化loss为0
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        self.loss.requires_grad_()

        
        with torch.no_grad():
            x = self.x
            t = self.t
            xb = self.xb
            tb = self.tb
            x0 = self.x0
            t0 = self.t0
            f = self.funF(x, t, self.nu)
            h = self.funU(xb, tb)
            q = self.funU(x0, t0)
            u_true = self.u_true

            
            us = self.net_model(x, t)
            self.A = self.Lus(us, x, t)

            ubs = self.net_model(xb, tb)
            u0s = self.net_model(x0, t0)
            
            A = torch.cat((self.A, ubs, u0s), dim=0)
            b = torch.cat((f, h, q), dim=0)

            W = self.lstsq(A=A, b=b)
            self.W = self.reload(W, requires_grad=False)

            # 打印耗时
            elapsed = time.time() - self.start_time
            log_str = 'Elapsed Time: %.4fs' % (elapsed)
            self.log(log_str)
            
            # rank = np.max(self.detach(torch.linalg.matrix_rank(us)))
            # self.log('rank ' + str(rank))


            u = torch.matmul(us, self.W)
            u_infty = np.max(self.detach(torch.linalg.norm(u_true-u, ord=inf)))
            u_rel_l2 = np.max(self.detach(torch.linalg.norm(u_true-u)/torch.linalg.norm(u_true)))           
            log_str = 'u_infty ' + str(u_infty) + ' u_rel_l2 ' + str(u_rel_l2)
            self.log(log_str)

            # 验证
            u_true = self.u_valid
            u = self.forward(self.x_valid, self.t_valid)
        
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
