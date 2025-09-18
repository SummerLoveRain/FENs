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
        x, y, xb, yb, x_valid, y_valid, self.a1, self.a2, self.k, self.funQ, self.funH, self.funU, self.N_basis = self.unzip_train_dict(
            train_dict=train_dict)

        # 区域内点
        self.x = self.data_loader(x)
        self.y = self.data_loader(y)
        # 边界点
        self.xb = self.data_loader(xb)
        self.yb = self.data_loader(yb)

        # 验证
        self.x_valid = self.data_loader(x_valid)
        self.y_valid = self.data_loader(y_valid)
        self.u_true = self.reload(self.funU(self.a1, self.a2, self.k, self.x, self.y), requires_grad=False)
        self.u_valid = self.reload(self.funU(self.a1, self.a2, self.k, self.x_valid, self.y_valid), requires_grad=False)

        self.W = None

        self.params = list(self.model.parameters())

        A = np.zeros([x.shape[0], self.N_basis])
        self.A = self.data_loader(A, requires_grad=False)

    def unzip_train_dict(self, train_dict):
        train_data = (
            train_dict['x'],
            train_dict['y'],
            train_dict['xb'],
            train_dict['yb'],
            train_dict['x_valid'],
            train_dict['y_valid'],
            train_dict['a1'],
            train_dict['a2'],
            train_dict['k'],
            train_dict['funQ'],
            train_dict['funH'],
            train_dict['funU'],
            train_dict['N_basis'],
        )
        return train_data

    def net_model(self, x, y):
        X = torch.cat((x, y), dim=1)
        X = self.coor_shift(X, self.lb, self.ub)
        us = self.model.forward(X)
        # us = torch.tanh(us)
        # us = self.model.act_func(us)
        return us

    def forward(self, x, y):
        us = self.net_model(x, y)
        u = torch.matmul(us, self.W)
        return u
    

    def lstsq(self, A, b):        
        W = torch.linalg.lstsq(A, b)[0]
        return W
    
    def diff_x(self, x, y, idx):    
        X = torch.cat((x, y), dim=1)
        X = self.coor_shift(X, self.lb, self.ub)
        shift_coeff = 2/(self.ub[idx]-self.lb[idx])
        us = self.model.diff_x(X, idx, shift_coeff)
        # us = torch.tanh(us)
        # us = self.model.act_func(us)
        return us
    
    def diff_xx(self, x, y, idx):
        X = torch.cat((x, y), dim=1)
        X = self.coor_shift(X, self.lb, self.ub)
        shift_coeff = 2/(self.ub[idx]-self.lb[idx])
        us = self.model.diff_xx(X, idx, shift_coeff)
        # us = torch.tanh(us)
        # us = self.model.act_func(us)
        return us
    
    def Lus(self, us, x, y):
        us_xx = self.diff_xx(x, y, idx=0)
        us_yy = self.diff_xx(x, y, idx=1)

        Lus = (us_xx + us_yy) + self.k**2 * us
        return Lus
    
    # 训练一次
    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        # 初始化loss为0
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        self.loss.requires_grad_()

        with torch.no_grad():
            
            # 区域点
            x = self.x
            y = self.y
            # 处理边界条件
            xb = self.xb
            yb = self.yb
            q = self.funQ(self.a1, self.a2, self.k, x, y)
            h = self.funH(self.a1, self.a2, self.k, xb, yb)
            u_true = self.u_true

            
            us = self.net_model(x, y)
            self.A = self.Lus(us, x, y)

            ubs = self.net_model(xb, yb)
            ubs = self.reload(ubs, requires_grad=False)
            
            A = torch.cat((self.A, ubs), dim=0)
            b = torch.cat((q, h), dim=0)

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
