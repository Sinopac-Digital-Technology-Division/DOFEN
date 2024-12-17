import torch
import torch.nn as nn
import numpy as np
import time

import datetime
import pickle
from torch.utils.data import DataLoader, Dataset, TensorDataset


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape(self.shape)
    
    
class FastGroupConv1d(nn.Conv1d):
    '''
    This class is built to resolve the issue: the slow operation speed of group convolution when number of groups are large.
    We found that directly using self-written matrix multiplication can dramatically accelerate operation speed of group convolution.
    The `fast_mode` argument decides when to switch from native operation in `nn.Conv1d` to self-written one by setting the group threshold.
    
    See following pytorch issues for more detailed description of this bug: 
    * https://github.com/pytorch/pytorch/issues/18631
    * https://github.com/pytorch/pytorch/issues/70954
    * https://github.com/pytorch/pytorch/issues/73764
    '''
    def __init__(self, *args, **kwargs):
        self.fast_mode = kwargs.pop('fast_mode')
        nn.Conv1d.__init__(self, *args, **kwargs)
        if self.groups > self.fast_mode:
            self.weight = nn.Parameter(
                self.weight.reshape(
                    self.groups, self.out_channels // self.groups, self.in_channels // self.groups, 1
                ).permute(3, 0, 2, 1)
            )
            self.bias = nn.Parameter(
                self.bias.unsqueeze(0).unsqueeze(-1)
            )
            
    def forward(self, x):
        if self.groups > self.fast_mode:
            x = x.reshape(-1, self.groups, self.in_channels // self.groups, 1)
            return (x * self.weight).sum(2, keepdims=True).permute(0, 1, 3, 2).reshape(-1, self.out_channels, 1) + self.bias
        else:
            return self._conv_forward(x, self.weight, self.bias)
    
    
class ConditionGeneration(nn.Module):
    def __init__(self, column_category_count, n_cond=128, categorical_optimized=False, fast_mode=64, device=torch.device('cpu')):
        super(ConditionGeneration, self).__init__()
        self.device = device
        self.fast_mode = fast_mode
        self.categorical_optimized = categorical_optimized
        
        self.num_index, self.cat_index, self.cat_count, self.cat_offset = self.get_num_cat_index(column_category_count)
        self.n_cond = n_cond
        self.phi_1 = self.get_phi_1()
        
    def get_num_cat_index(self, column_category_count):
        num_index = []
        cat_index = []
        cat_count = []
        for idx, ele in enumerate(column_category_count):
            if ele == -1:
                num_index.append(idx)
            else:
                cat_index.append(idx)
                cat_count.append(ele)
        cat_offset = torch.tensor([0] + np.cumsum(cat_count).tolist()[:-1]).long().to(self.device)
        return num_index, cat_index, cat_count, cat_offset

    def get_phi_1(self, ):
        phi_1 = nn.ModuleDict()
        if len(self.num_index):
            phi_1['num'] = nn.Sequential(
                # input = (b, n_num_col)
                # output = (b, n_num_col, n_cond)
                Reshape(-1, len(self.num_index), 1),
                FastGroupConv1d(len(self.num_index), len(self.num_index)*self.n_cond, kernel_size=1, groups=len(self.num_index), fast_mode=self.fast_mode),
                nn.Sigmoid(),
                Reshape(-1, len(self.num_index), self.n_cond)
            )
        if len(self.cat_index):
            phi_1['cat'] = nn.ModuleDict()
            phi_1['cat']['embedder'] = nn.Embedding(sum(self.cat_count), self.n_cond)            
            phi_1['cat']['mapper'] = nn.Sequential(
                # input = (b, n_cat_col, n_cond)
                # output = (b, n_cat_col, n_cond)
                Reshape(-1, len(self.cat_index) * self.n_cond, 1),
                nn.GroupNorm(len(self.cat_index), len(self.cat_index) * self.n_cond),
                FastGroupConv1d(len(self.cat_index)*self.n_cond, len(self.cat_index)*self.n_cond, kernel_size=1, groups=len(self.cat_index)*self.n_cond if self.categorical_optimized else len(self.cat_index), fast_mode=self.fast_mode),                
                nn.Sigmoid(),
                Reshape(-1, len(self.cat_index), self.n_cond)
            ) 
        return phi_1
    
    def forward(self, x):
        M = []
        
        if len(self.num_index):
            num_x = x[:, self.num_index].float()
            num_sample_emb = self.phi_1['num'](num_x)
            M.append(num_sample_emb)
            
        if len(self.cat_index):
            cat_x = x[:, self.cat_index].long() + self.cat_offset
            cat_sample_emb = self.phi_1['cat']['mapper'](self.phi_1['cat']['embedder'](cat_x))
            M.append(cat_sample_emb)
            
        M = torch.cat(M, dim=1) # (b, n_col, n_cond)
        M = M.permute(0, 2, 1) # (b, n_cond, n_col)
        return M
    
    
class rODTConstruction(nn.Module):
    def __init__(self, n_cond, n_col):
        super().__init__()
        self.permutator = torch.rand(n_cond * n_col).argsort(-1)
    
    def forward(self, M):
        return M.reshape(M.shape[0], -1, 1)[:, self.permutator, :]
    
    
class rODTForestConstruction(nn.Module):
    def __init__(self, n_col, n_rodt, n_cond, n_estimator, n_head=1, n_hidden=128, n_forest=100, dropout=0.0, fast_mode=64, device=torch.device('cpu')):
        super().__init__()
        
        self.device = device
        self.n_estimator = n_estimator
        self.n_forest = n_forest
        self.n_rodt = n_rodt
        self.n_head = n_head
        self.n_hidden = n_hidden
        
        self.phi_2 = nn.Sequential(
            nn.GroupNorm(n_rodt, n_cond * n_col),
            nn.Dropout(dropout),
            FastGroupConv1d(n_cond * n_col, n_cond * n_col, groups=n_rodt, kernel_size=1, fast_mode=fast_mode),
            nn.ReLU(),
            nn.GroupNorm(n_rodt, n_cond * n_col),
            nn.Dropout(dropout),
            FastGroupConv1d(n_cond * n_col, n_rodt * n_head, groups=n_rodt, kernel_size=1, fast_mode=fast_mode), 
            Reshape(-1, n_rodt, n_head)
        )
        self.E = nn.Embedding(n_rodt, n_hidden)
        self.sample_without_replacement_eval = self.get_sample_without_replacement()
        
    def get_sample_without_replacement(self, ):
        return torch.rand(self.n_forest, self.n_rodt, device=self.device).argsort(-1)[:, :self.n_estimator]
        
    def forward(self, O):
        b = O.shape[0]
        w = self.phi_2(O) # (b, n_rodt, n_head)
        E = self.E.weight.unsqueeze(0) # (1, n_rodt, n_hidden)
        
        sample_without_replacement = self.get_sample_without_replacement() if self.training else self.sample_without_replacement_eval

        w_prime = w[:, sample_without_replacement].softmax(-2).unsqueeze(-1) # (b, n_forest, n_rodt, n_head, 1)
        E_prime = E[:, sample_without_replacement].reshape(
            1, self.n_forest, self.n_estimator, self.n_head, self.n_hidden // self.n_head
        ) # (1, n_forest, n_rodt, n_head, n_hidden // n_head)
        F = (w_prime * E_prime).sum(-3).reshape(
            b, self.n_forest, self.n_hidden
        ) # (b, n_forest, n_hidden)
        return F
    
    
class rODTForestBagging(nn.Module):
    def __init__(self, n_hidden, dropout, n_class):
        super().__init__()
        self.phi_3 = nn.Sequential(
            nn.LayerNorm(n_hidden),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.LayerNorm(n_hidden),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_class)
        )
    
    def forward(self, F):
        y_hat = self.phi_3(F) # (b, n_forest, n_class) 
        return y_hat
    
    
class DOFEN(nn.Module):
    def __init__(
        self, 
        column_category_count, 
        n_class, 
        
        m=16, 
        d=4, 
        n_head=1,
        n_forest=100,
        n_hidden=128,
        dropout=0.0, 
        
        ### experimental functionality ###
        categorical_optimized=False,
        fast_mode=2048,
        use_bagging_loss=False,
        ### ###
        
        device=torch.device('cpu'),
        verbose=False
    ):
        super().__init__()
        
        self.device = device
        self.n_class = 1 if n_class == -1 else n_class
        self.is_rgr = True if n_class == -1 else False
        
        self.m = m
        self.d = d
        self.n_head = n_head
        self.n_forest = n_forest
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.use_bagging_loss = use_bagging_loss
        
        self.n_cond = self.d * self.m
        self.n_col = len(column_category_count)
        self.n_rodt = self.n_cond * self.n_col // self.d
        self.n_estimator = max(2, int(self.n_col ** 0.5)) * self.n_cond // self.d
        
        self.condition_generation = ConditionGeneration(            
            column_category_count, 
            n_cond=self.n_cond, 
            categorical_optimized=categorical_optimized, 
            fast_mode=fast_mode,
            device=self.device
        )
        self.rodt_construction = rODTConstruction(
            self.n_cond, 
            self.n_col
        )
        self.rodt_forest_construction = rODTForestConstruction(
            self.n_col, 
            self.n_rodt, 
            self.n_cond,
            self.n_estimator,
            n_head=self.n_head, 
            n_hidden=self.n_hidden,
            n_forest=self.n_forest,
            dropout=self.dropout,
            fast_mode=fast_mode,
            device=self.device
        )
        self.rodt_forest_bagging = rODTForestBagging(
            self.n_hidden,
            self.dropout,
            self.n_class
        )
        
        if verbose:
            print('='*20)
            print('total condition: ', self.n_cond * self.n_col)
            print('n_rodt: ', self.n_rodt)
            print('n_estimator: ', self.n_estimator)    
            print('='*20)    
        
    def calc_loss(self, y_hat, y):
        if self.is_rgr:
            loss = torch.nn.functional.mse_loss(y_hat.squeeze(-1), y.float())
        else:
            loss = torch.nn.functional.cross_entropy(y_hat, y.long())
        return loss
    
    def timer(self, x):
        self.eval()
        x = x.to(self.device)
        
        times = []
        
        times.append(time.perf_counter())
        M = self.condition_generation(x) # (b, n_cond, n_col)
        times.append(time.perf_counter())
        O = self.rodt_construction(M) # (b, n_rodt, d)
        times.append(time.perf_counter())
        F = self.rodt_forest_construction(O) # (b, n_forest, n_hidden)
        times.append(time.perf_counter())
        y_hat = self.rodt_forest_bagging(F) # (b, n_forest, n_class)
        times.append(time.perf_counter())
        y_hat_final = y_hat.detach().mean(1) # (b, n_class)
        times.append(time.perf_counter())
        
        self.train()
        
        times = np.array(times)
        return times[1:] - times[:-1]

    def forward(self, x, y=None):
        x = x.to(self.device)
        
        M = self.condition_generation(x) # (b, n_cond, n_col)
        O = self.rodt_construction(M) # (b, n_rodt, d) 
        F = self.rodt_forest_construction(O) # (b, n_forest, n_hidden)
        y_hat = self.rodt_forest_bagging(F) # (b, n_forest, n_class)
        y_hat_final = y_hat.detach().mean(1) # (b, n_class)
        
        if y is not None:
            y = y.to(self.device)
            loss = self.calc_loss(
                y_hat.permute(0, 2, 1) if not self.is_rgr else y_hat, 
                y.unsqueeze(-1).expand(-1, self.n_forest)
            )
            if self.n_forest > 1 and self.training and self.use_bagging_loss:
                loss += self.calc_loss(y_hat.mean(1), y)
        else:
            loss = torch.tensor(0.0)
       
        return y_hat_final, loss
    
    
class DOFENTrainer():
    """This is the training and inference interface of DOFEN
    
    Args:
        dofen_config (dict): 
            ### standard usage ###
            - column_category_count (list of int): number of categories for each column, set the value to -1 for numerical columns
            - n_class (int): number of class of a dataset, please set to 2 for binary tasks, set to 'number of class' for multiclass tasks, and set to -1 for regression tasks 
            - m (int): an intermediate parameter ensures that number of rODT is an integer, larger m result in more rODTs, search space = [16, 32, 64]
            - d (int): depth of a rODT, search space = [3, 4, 6, 8] 
            - n_forest (int): number of rODT forest generated for forest ensemble
            - n_hidden (int): hidden dimension of rODT embedding
            - dropout (float): dropout rate, search space = [0.0, 0.1, 0.2]
            - device (torch.device): torch device, Sets the device for tensors initialized in the forward function, accelerating computations by placing them directly on the desired device (e.g. GPU). Note that this setting only affects tensors created in forward, and should match the device used by the DOFEN model.
            - verbose (bool): whether to print model configuration when initialize

            ### advanced usage, These functionalities strengthen DOFEN's performance and efficiency, but are not implemented in the paper of NeurIPS 2024 version ###
            - n_head (int): A multi-head extension of "Two-level Relaxed Forest Ensemble", increase number of heads (e.g. 4) greatly improves performance on larger datasets (e.g. n_samples > 10000), default = 1, search space = [1, 4, 8]
            - categorical_optimized (bool): A simpler encoding layer for categorical columns, when set to True, model uses less parameters but improves performance, default = False.
            - fast_mode (int): A faster version of group convolution when having large number of groups (i.e. number of rODTs in DOFEN), will start to use the faster version if number of groups is larger than the set value, default = 32.
            - use_bagging_loss (bool): DOFEN default calculate loss individually for each tree, when set ot True, an additional loss is calculated on the ensemble prediction, default = False.

        train_config (dict): 
            - batch_size (int): Number of batch size.
            - n_epochs (int): Number of training epochs.
            - early_stopping_patience (int): Training is early stopped if model is not improving on validation set for this many of epochs, DOFEN originally does not use early stopping, default = -1, set this number larger than 0 if you want to early stop.
            - save_dir (str): Model save path if early stopping is used.
            - ckpt_name (str): Model checkpoint name, model will be saved when validation performance improve.

        eval_config  (dict):
            - metric (dict): dictionary containing evaluation metrics
                - classification (function): Evaluation metric for classification task, default = sklearn.metric.accuracy_score
                - regression (function): Evaluation metric for classification task, default = sklearn.metric.r2_score

    Returns:
        None
    """

    def __init__(self, dofen_config, train_config, eval_config):
        self.dofen_config = dofen_config
        self.batch_size = train_config['batch_size']
        self.n_epochs = train_config['n_epochs']
        self.early_stopping_patience = train_config['early_stopping_patience']
        self.save_dir = train_config['save_dir']
        self.ckpt_name = train_config['ckpt_name']
        self.eval_metric = eval_config['metric']
        
    def set_seed(self, torch_seed=0, deterministic=True):
        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
        torch.cuda.manual_seed(torch_seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        
    def init(self, ):
        self.set_seed(0)
        self.model = DOFEN(**self.dofen_config).to(self.dofen_config['device'])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.0)
        
    def fit(self, tr_x, tr_y, va_x=None, va_y=None, te_x=None, te_y=None):
        self.tr_dataloader = DataLoader(
            TensorDataset(torch.tensor(tr_x), torch.tensor(tr_y)), 
            batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        
        if va_x is not None and va_y is not None:
            self.va_dataloader = DataLoader(
                TensorDataset(torch.tensor(va_x), torch.tensor(va_y)), 
                batch_size=self.batch_size, shuffle=False, drop_last=False
            )
            best_perf = -np.inf
            best_epoch = -1
            no_improve = 0
        else:
            self.va_dataloader = None
            
        if te_x is not None and te_y is not None:
            self.te_dataloader = DataLoader(
                TensorDataset(torch.tensor(te_x), torch.tensor(te_y)), 
                batch_size=self.batch_size, shuffle=False, drop_last=False
            )
        else:
            self.te_dataloader = None
        
        for epoch in range(self.n_epochs):
            self.model.train()
            print(f'Epoch: {epoch+1} | {datetime.datetime.now()}')
            
            for x, y in self.tr_dataloader:
                _, loss = self.model(x, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            if self.va_dataloader is not None and self.early_stopping_patience > 0:
                perf_name, curr_perf = self.evaluate_with_dataloader(self.va_dataloader)
                if curr_perf > best_perf:
                    best_perf = curr_perf
                    best_epoch = epoch + 1
                    no_improve = 0
                    torch.save(self.model, f'{self.save_dir}/{self.ckpt_name}.ckpt')
                else:
                    no_improve += 1
                    print(f'Performance not improve for {no_improve} epochs, best epoch is {best_epoch}')
                    
                if no_improve >= self.early_stopping_patience:
                    break
                    
            if self.te_dataloader is not None:
                perf_name, curr_perf = self.evaluate_with_dataloader(self.te_dataloader)
                print(f'testing {perf_name}: {curr_perf}')

    def evaluate_with_dataloader(self, dataloader):
        self.model.eval()
        preds = []
        ys = []
        with torch.no_grad():
            for x, y in dataloader:
                pred, loss = self.model(x, y)
                preds.append(pred.cpu())
                ys.append(y.cpu())     
        preds = torch.cat(preds).numpy()
        ys = torch.cat(ys).numpy()
        if self.model.is_rgr:
            curr_perf = self.eval_metric['regression'](ys, preds.squeeze())
            perf_name = self.eval_metric['regression'].__name__
        else:
            curr_perf = self.eval_metric['classification'](ys, preds.argmax(-1))
            perf_name = self.eval_metric['classification'].__name__
        return perf_name, curr_perf

    def predict(self, x, eval_batch_size=256):
        x = torch.tensor(x)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for chunk_x in x.chunk(max(1, x.shape[0] // eval_batch_size)):
                preds.append(self.model(chunk_x)[0].cpu())
        preds = torch.cat(preds).numpy()
        return preds
        
    def evaluate(self, x, y, eval_batch_size=256):
        pred = self.predict(x, eval_batch_size=eval_batch_size)
        if self.model.is_rgr:
            curr_perf = self.eval_metric['regression'](y, pred.squeeze())
            perf_name = self.eval_metric['regression'].__name__
        else:
            curr_perf = self.eval_metric['classification'](y, pred.argmax(-1))
            perf_name = self.eval_metric['classification'].__name__
        print(f'{perf_name}: {curr_perf}')
        