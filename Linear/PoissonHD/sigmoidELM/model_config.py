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
        x, x_b, self.funF, self.funU, self.N_basis = self.unzip_train_dict(
            train_dict=train_dict)
        
        self.d = x.shape[1]

        # 区域内点
        self.x = []
        for i in range(self.d):
            xi = x[:, i:i+1]
            X = self.data_loader(xi)
            self.x.append(X)

        self.x_b = []
        for i in range(self.d):
            xi = x_b[:, i:i+1]
            X = self.data_loader(xi)
            self.x_b.append(X)
       
        self.u_true = self.data_loader(self.funU(self.x), requires_grad=False)

        self.W = None
        self.params = list(self.model.parameters())


        A = np.zeros([x.shape[0], self.N_basis])
        self.A = self.data_loader(A, requires_grad=False)

    def unzip_train_dict(self, train_dict):
        train_data = (
            train_dict['x'],
            train_dict['x_b'],
            train_dict['funF'],
            train_dict['funU'],
            train_dict['N_basis'],
        )
        return train_data

    def net_model(self, *args):
        X = torch.cat(*args, dim=1)
        X = self.coor_shift(X, self.lb, self.ub)
        
        us = self.model.forward(X)
        # us = torch.tanh(us)
        
        return us

    def forward(self, *args):
        us= self.net_model(*args)
      
        u = torch.matmul(us, self.W)
        return u
    
    def lstsq(self, A, b):        
        W = torch.linalg.lstsq(A, b)[0]
        return W
    
    
    def diff_x(self, X, idx):
        X = self.coor_shift(X, self.lb, self.ub)
        shift_coeff = 2/(self.ub[idx]-self.lb[idx])
        us = self.model.diff_x(X, idx, shift_coeff)
        # us = torch.tanh(us)
        # us = self.model.act_func(us)
        return us
    
    def diff_xx(self, X, idx):
        X = self.coor_shift(X, self.lb, self.ub)
        shift_coeff = 2/(self.ub[idx]-self.lb[idx])
        us = self.model.diff_xx(X, idx, shift_coeff)
        # us = torch.tanh(us)
        # us = self.model.act_func(us)
        return us
    
    def Lus(self, *args):    
        X = torch.cat(*args, dim=1)
        d = X.shape[1]
        Lus = 0
        for i in range(d):
            us_xx = self.diff_xx(X, idx=i)
            Lus -= us_xx
        return Lus
    
    def Lu(self, u, X):
        d = len(X)
        Lu = 0
        for i in range(d):
            x = X[i]
            u_x = self.compute_grad(u, x)
            u_xx = self.compute_grad(u_x, x)
            Lu -= u_xx
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
            x_b = self.x_b
            f = self.funF(x)
            h = self.funU(x_b)
            u_true = self.u_true

            d = len(x)
            N_batch = 10
            batch_size = x[0].shape[0]//N_batch
            res = x[0].shape[0]%N_batch
            for j in range(N_batch+res):
                start_time = time.time()
                m = j*batch_size
                n = (j+1)*batch_size
                x_t = []
                for i in range(d):
                    x_t.append(x[i][m:n,:])

                Lus = self.Lus(x_t)
                self.A[j*batch_size:(j+1)*batch_size, :] = Lus
                    
                elapsed = time.time() - start_time
                log_str = 'Time: %.4fs for %d -th batch' % (elapsed, (j+1))
                self.log(log_str)
            
            ubs_D = self.net_model(x_b)
            
            A = torch.cat((self.A, ubs_D), dim=0)
            b = torch.cat((f, h), dim=0)

            W = self.lstsq(A=A, b=b)
            self.W = self.reload(W, requires_grad=False)

            # 打印耗时
            elapsed = time.time() - self.start_time
            log_str = 'Elapsed Time: %.4fs' % (elapsed)
            self.log(log_str)
            
            # rank = np.max(self.detach(torch.linalg.matrix_rank(A)))
            # self.log('rank ' + str(rank))


            u = self.forward(x)
            u_infty = np.max(self.detach(torch.linalg.norm(u_true-u, ord=inf)))
            u_rel_l2 = np.max(self.detach(torch.linalg.norm(u_true-u)/torch.linalg.norm(u_true)))           
            log_str = 'u_infty ' + str(u_infty) + ' u_rel_l2 ' + str(u_rel_l2)
            self.log(log_str)

            u = self.forward(x_b)
            u_true = self.funU(x_b)
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
