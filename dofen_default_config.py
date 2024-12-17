import torch
from sklearn.metrics import accuracy_score, r2_score

dofen_config = {
    'm': 16, 
    'd': 4, 
    'n_head': 1,
    'n_forest': 100,
    'n_hidden': 128,
    'dropout': 0.0, 

    'categorical_optimized': False,
    'fast_mode': 32, 
    'use_bagging_loss': False,

    'device': torch.device('cuda:0'),
    'verbose': True
}
    
train_config = {
    'batch_size': 256,
    'n_epochs': 500,
    'early_stopping_patience': -1,
    'save_dir': './',
    'ckpt_name': 'best'
}

eval_config = {
    'metric': {
        'classification': accuracy_score,
        'regression': r2_score
    }
}