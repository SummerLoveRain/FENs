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
        x, xb, self.lamb, self.beta, self.funF, self.funU, x_valid, self.N_basis = self.unzip_train_dict(
            train_dict=train_dict)

        # 区域内点
        self.x = self.data_loader(x)
        self.x_valid = self.data_loader(x_valid)
        # 边界点
        self.xb = self.data_loader(xb)

        self.u_true = self.reload(self.funU(self.x), requires_grad=False)
        self.u_valid = self.reload(self.funU(self.x_valid), requires_grad=False)

        self.W = None
        self.params = list(self.model.parameters())

        A = np.zeros([x.shape[0], self.N_basis])
        self.A = self.data_loader(A, requires_grad=False)

    def unzip_train_dict(self, train_dict):
        train_data = (
            train_dict['x'],
            train_dict['xb'],
            train_dict['lamb'],
            train_dict['beta'],
            train_dict['funF'],
            train_dict['funU'],
            train_dict['x_valid'],
            train_dict['N_basis'],
        )
        return train_data

    def net_model(self, x):
        X = x
        X = self.coor_shift(X, self.lb, self.ub)
        us = self.model.forward(X)
        # us = self.model.act_func(us)
        return us

    def forward(self, x):
        us = self.net_model(x)
        u = torch.matmul(us, self.W)
        return u

    def lstsq(self, A, b):        
        W = torch.linalg.lstsq(A, b)[0]
        return W
    
    
    def diff_x(self, x, idx):    
        X = x
        X = self.coor_shift(X, self.lb, self.ub)
        shift_coeff = 2/(self.ub[idx]-self.lb[idx])
        us = self.model.diff_x(X, idx, shift_coeff)
        # us = torch.tanh(us)
        # us = self.model.act_func(us)
        return us
    
    def diff_xx(self, x, idx):
        X = x
        X = self.coor_shift(X, self.lb, self.ub)
        shift_coeff = 2/(self.ub[idx]-self.lb[idx])
        us = self.model.diff_xx(X, idx, shift_coeff)
        # us = torch.tanh(us)
        # us = self.model.act_func(us)
        return us
    
    def Lus(self, us, x):
        us_xx = self.diff_xx(x, idx=0)

        Lus = us_xx - self.lamb*us
        return Lus
    
    def Lu(self, u, x):
        u_x = self.compute_grad(u, x)
        u_xx = self.compute_grad(u_x, x)

        Lu = u_xx - self.lamb * u
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
            # 区域点
            x = self.x
            # 处理边界条件
            xb = self.xb
            f = self.funF(self.lamb, self.beta, x)
            h = self.funU(xb)
            u_true = self.u_true
            
            us = self.net_model(x)
            self.A = self.Lus(us, x)


            ubs = self.net_model(xb)
            ubs = self.reload(ubs, requires_grad=False)
            A = torch.cat((self.A, ubs), dim=0)

            Wk = self.reload(self.xavier_init([self.N_basis, 1]), requires_grad=False)
            # us = self.net_model(x)
            us = self.reload(us, requires_grad=False)
            for k in range(100):
                # log_str = 'Inner_Iter ' + str(k+1)
                # self.log(log_str)
                # 初始化loss为0
                self.optimizer.zero_grad()

                uk = torch.matmul(us, Wk)
                N_uk = self.beta*torch.sin(uk)

                b = torch.cat((f-N_uk, h), dim=0)
                # b = torch.cat((f, h), dim=0)

                W = self.lstsq(A=A, b=b)
                self.W = self.reload(W, requires_grad=False)
                
                
                # u = torch.matmul(us, W)
                # u_true = self.u_true
                # u_infty = np.max(self.detach(torch.linalg.norm(u_true-u, ord=inf)))
                # u_rel_l2 = np.max(self.detach(torch.linalg.norm(u_true-u)/torch.linalg.norm(u_true)))           
                # log_str = 'u_infty ' + str(u_infty) + ' u_rel_l2 ' + str(u_rel_l2)
                # self.log(log_str)

                # # 验证
                # u = self.forward(self.x_valid)
                # u_true = self.u_valid
                # u_infty = np.max(self.detach(torch.linalg.norm(u_true-u, ord=inf)))
                # u_rel_l2 = np.max(self.detach(torch.linalg.norm(u_true-u)/torch.linalg.norm(u_true)))           
                # log_str = 'u_infty ' + str(u_infty) + ' u_rel_l2 ' + str(u_rel_l2)
                # self.log(log_str)

                # Aw = torch.matmul(A, W)
                # loss_infty = np.max(self.detach(torch.linalg.norm(b - Aw, ord=inf)))
                # loss_rel_l2 = np.max(self.detach(torch.linalg.norm(b - Aw)/torch.linalg.norm(b)))
                # log_str = 'loss_infty ' + str(loss_infty) + ' loss_rel_l2 ' + str(loss_rel_l2)
                # self.log(log_str)

                
                W_infty = np.max(self.detach(torch.linalg.norm(W - Wk, ord=inf)))
                # W_rel_l2 = np.max(self.detach(torch.linalg.norm(W - Wk)/torch.linalg.norm(W)))
                # log_str = 'W_infty ' + str(W_infty) + ' W_rel_l2 ' + str(W_rel_l2)
                # self.log(log_str)
                
                if W_infty < 1e-16:
                    break
                Wk = W

            # 打印耗时
            elapsed = time.time() - self.start_time
            log_str = 'Elapsed Time: %.4fs' % (elapsed)
            self.log(log_str)
            
            # rank = np.max(self.detach(torch.linalg.matrix_rank(us)))
            # self.log('rank ' + str(rank))

            u_true = self.u_true
            u = self.forward(self.x)
            u_infty = np.max(self.detach(torch.linalg.norm(u_true-u, ord=inf)))
            u_rel_l2 = np.max(self.detach(torch.linalg.norm(u_true-u)/torch.linalg.norm(u_true)))           
            log_str = 'u_infty ' + str(u_infty) + ' u_rel_l2 ' + str(u_rel_l2)
            self.log(log_str)

            # 验证
            u_true = self.u_valid
            u = self.forward(self.x_valid)
        
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
