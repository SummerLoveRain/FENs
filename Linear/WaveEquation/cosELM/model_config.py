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
        x, y, t, xb, yb, tb, x0, y0, t0, x_valid, y_valid, t_valid, self.funF, self.funW, self.funU, self.N_basis = self.unzip_train_dict(
            train_dict=train_dict)

        # 区域内点
        self.x = self.data_loader(x)
        self.y = self.data_loader(y)
        self.t = self.data_loader(t)
        self.xb = self.data_loader(xb)
        self.yb = self.data_loader(yb)
        self.tb = self.data_loader(tb)
        self.x0 = self.data_loader(x0)
        self.y0 = self.data_loader(y0)
        self.t0 = self.data_loader(t0)

        self.u_true = self.data_loader(self.funU(x, y, t, type='np'), requires_grad=False)
        self.u_valid = self.data_loader(self.funU(x_valid, y_valid, t_valid, type='np'), requires_grad=False)
        self.x_valid = self.data_loader(x_valid, requires_grad=False)
        self.y_valid = self.data_loader(y_valid, requires_grad=False)
        self.t_valid = self.data_loader(t_valid, requires_grad=False)

        self.W = None

        A = np.zeros([x.shape[0], self.N_basis])
        self.A = self.data_loader(A, requires_grad=False)
        A0 = np.zeros([x0.shape[0], self.N_basis])
        self.A0 = self.data_loader(A0, requires_grad=False)

    def unzip_train_dict(self, train_dict):
        train_data = (
            train_dict['x'],
            train_dict['y'],
            train_dict['t'],
            train_dict['xb'],
            train_dict['yb'],
            train_dict['tb'],
            train_dict['x0'],
            train_dict['y0'],
            train_dict['t0'],
            train_dict['x_valid'],
            train_dict['y_valid'],
            train_dict['t_valid'],
            train_dict['funF'],
            train_dict['funW'],
            train_dict['funU'],
            train_dict['N_basis'],
        )
        return train_data

    def net_model(self, x, y, t):
        X = torch.cat((x, y, t), dim=1)
        # X = self.coor_shift(X, self.lb, self.ub)
        us = self.model.forward(X)
        # us = torch.tanh(us)
        # us = self.model.act_func(us)
        return us

    def forward(self, x, y, t):
        us = self.net_model(x, y, t)
        u = torch.matmul(us, self.W)
        return u
    
    def lstsq(self, A, b):        
        W = torch.linalg.lstsq(A, b)[0]
        return W
    
    
    def diff_x(self, x, y, t, idx):    
        X = torch.cat((x, y, t), dim=1)
        # X = self.coor_shift(X, self.lb, self.ub)
        us = self.model.diff_x(X, idx)
        # us = torch.tanh(us)
        # us = self.model.act_func(us)
        return us
    
    def diff_xx(self, x, y, t, idx):
        X = torch.cat((x, y, t), dim=1)
        # X = self.coor_shift(X, self.lb, self.ub)
        us = self.model.diff_xx(X, idx)
        # us = torch.tanh(us)
        # us = self.model.act_func(us)
        return us
    
    def Lus(self, x, y, t):
        us_tt = self.diff_xx(x, y, t, idx=2)
        us_xx = self.diff_xx(x, y, t, idx=0)
        us_yy = self.diff_xx(x, y, t, idx=1)

        Lus = us_tt - (us_xx + us_yy)
        return Lus


    def Lu(self, u, x, y, t):
        u_t = self.compute_grad(u, t)
        u_tt = self.compute_grad(u_t, t)
        u_x = self.compute_grad(u, x)
        u_xx = self.compute_grad(u_x, x)
        u_y = self.compute_grad(u, y)
        u_yy = self.compute_grad(u_y, y)
        Lu = u_tt - (u_xx + u_yy)
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
            y = self.y
            t = self.t
            xb = self.xb
            yb = self.yb
            tb = self.tb
            x0 = self.x0
            y0 = self.y0
            t0 = self.t0
            f = self.funF(x, y, t)
            h = self.funU(xb, yb, tb)
            q = self.funU(x0, y0, t0)
            w = self.funW(x0, y0, t0)
            u_true = self.u_true

            
            N_batch = 2
            batch_size = x.shape[0]//N_batch
            batch_size1 = x0.shape[0]//N_batch
            for j in range(50):
                start_time = time.time()
                x_t = x[j*batch_size:(j+1)*batch_size, :]
                y_t = y[j*batch_size:(j+1)*batch_size, :]
                t_t = t[j*batch_size:(j+1)*batch_size, :]
                x0_t = x0[j*batch_size1:(j+1)*batch_size1, :]
                y0_t = y0[j*batch_size1:(j+1)*batch_size1, :]
                t0_t = t0[j*batch_size1:(j+1)*batch_size1, :]

                Lus = self.Lus(x_t, y_t, t_t)
                self.A[j*batch_size:(j+1)*batch_size, :] = Lus

                us0_t = self.diff_x(x0_t, y0_t, t0_t, idx=2)
                self.A0[j*batch_size1:(j+1)*batch_size1, :] = us0_t

                elapsed = time.time() - start_time
                log_str = 'Time: %.4fs for %d -th batch' % (elapsed, (j+1))
                self.log(log_str)


            ubs = self.net_model(xb, yb, tb)
            ubs = self.reload(ubs, requires_grad=False)
            u0s = self.net_model(x0, y0, t0)
            u0s = self.reload(u0s, requires_grad=False)
                        
            Lus = self.A
            A0 = self.A0
            A = torch.cat((Lus, ubs, u0s, A0), dim=0)
            b = torch.cat((f, h, q, w), dim=0)

            W = self.lstsq(A=A, b=b)
            self.W = self.reload(W, requires_grad=False)

            # 打印耗时
            elapsed = time.time() - self.start_time
            log_str = 'Elapsed Time: %.4fs' % (elapsed)
            self.log(log_str)
            
            # rank = np.max(self.detach(torch.linalg.matrix_rank(us)))
            # self.log('rank ' + str(rank))


            u = self.forward(x, y, t)
            u_infty = np.max(self.detach(torch.linalg.norm(u_true-u, ord=inf)))
            u_rel_l2 = np.max(self.detach(torch.linalg.norm(u_true-u)/torch.linalg.norm(u_true)))           
            log_str = 'u_infty ' + str(u_infty) + ' u_rel_l2 ' + str(u_rel_l2)
            self.log(log_str)

            # 验证
            u_true = self.u_valid
            u = self.forward(self.x_valid, self.y_valid, self.t_valid)
        
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
