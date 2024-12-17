from generate_dataset_pipeline import generate_dataset
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pickle
import os


CONFIG_DEFAULT = {
    'LARGE': {
        'data__method_name': 'openml_no_transform',
        'train_prop': 0.70,
        'val_test_prop': 0.3,
        'max_train_samples': 50000,
        'max_val_samples': 50000,
        'max_test_samples': 50000,
        'transform__0__apply_on': 'numerical',
        'transform__0__method_name': 'gaussienize',
        'transform__0__type': 'quantile'
    },
    'MED': {
        'data__method_name': 'openml_no_transform',
        'train_prop': 0.70,
        'val_test_prop': 0.3,
        'max_train_samples': 10000,
        'max_val_samples': 50000,
        'max_test_samples': 50000,
        'transform__0__apply_on': 'numerical',
        'transform__0__method_name': 'gaussienize',
        'transform__0__type': 'quantile'
    }
}

CONFIG_RAW_X = {
    'LARGE': {
        'data__method_name': 'openml_no_transform',
        'train_prop': 0.70,
        'val_test_prop': 0.3,
        'max_train_samples': 50000,
        'max_val_samples': 50000,
        'max_test_samples': 50000,
    },
    'MED': {
        'data__method_name': 'openml_no_transform',
        'train_prop': 0.70,
        'val_test_prop': 0.3,
        'max_train_samples': 10000,
        'max_val_samples': 50000,
        'max_test_samples': 50000,
    }
}


BENCHMARKS = {
    'MED': {
        'CAT_REG': {
            'data__keyword': [
                361093,361094,361096,361097,361098,361099,361101,361102,361103,361104,361287,
                361288,361289,361291,361292,361293,361294,
            ],
            'data__categorical': True,
            'data__regression': True,
            'regression': True,
        },    
        'NUM_REG': {
            'data__keyword': [
                361072,361073,361074,361076,361077,361078,361079,361080,361081,361082,361083,
                361084,361085,361086,361087,361088,361279,361280,361281
            ],
            'data__categorical': False,
            'data__regression': True,
            'regression': True,
        },
        'CAT_CLS': {
            'data__keyword': [
                361110,361111,361113,361282,361283,361285,361286
            ],
            'data__categorical': True,
            'data__regression': False,
            'regression': False,
        },
        'NUM_CLS' : {
            'data__keyword': [
                361055,361060,361061,361062,361063,361065,361066,361068,361069,361070,361273,361274,
                361275,361276,361277,361278
            ],
            'data__categorical': False,
            'data__regression': False,
            'regression': False,
        }
    },
    'LARGE': {
        'CAT_REG': {
            'data__keyword': [
                361095, 361096, 361101, 361103, 361104
            ],
            'data__categorical': True,
            'data__regression': True,
            'regression': True,
        },
        'NUM_REG': {
            'data__keyword': [
                361091, 361080, 361083
            ],
            'data__categorical': False,
            'data__regression': True,
            'regression': True,
        },
        'CAT_CLS': {
            'data__keyword': [
                361113, 361285
            ],
            'data__categorical': True,
            'data__regression': False,
            'regression': False,
        },
        'NUM_CLS': {
            'data__keyword': [
                361061, 361068, 361069, 361274, 
            ],
            'data__categorical': False,
            'data__regression': False,
            'regression': False,
        }
    }    
}


def calc_col_cat_count(x_train, x_val, x_test, categorical_indicator):
    col_cat_count = []
    if categorical_indicator is not None:
        for idx, ele in enumerate(categorical_indicator):
            if not ele:
                col_cat_count.append(-1)
            else:
                col_cat_count.append(len(np.unique(np.concatenate([x_train[:, idx], x_val[:, idx], x_test[:, idx]]))))
    else:
        col_cat_count = [-1 for _ in range(x_train.shape[1])]              
    return col_cat_count


def calc_label_cat_count(y_train, y_val, y_test, is_rgr):
    label_cat_count = []
    if is_rgr:
        label_cat_count.append(-1)
    else:
        label_cat_count.append(len(np.unique(np.concatenate([y_train, y_val, y_test]).reshape(-1))))
    return label_cat_count[0]



if __name__ == '__main__':
    main_dir = {
        'MED': '../tabular_benchmark_data',
        'LARGE': '../tabular_benchmark_data_large'
    }

    for benchmark_size in ['MED', 'LARGE']:
        for benchmark_name, benchmark_config in BENCHMARKS[benchmark_size].items():
            for benchmark_dataset in benchmark_config['data__keyword']:
                print(f'===== data: {benchmark_dataset} =====')

                ### build config ###
                dataset_config = {
                    **CONFIG_DEFAULT[benchmark_size],
                    **benchmark_config, 
                    **{
                        'data__keyword': benchmark_dataset, 
                    }
                }
                dataset_config_rawx = {
                    **CONFIG_RAW_X[benchmark_size],
                    **benchmark_config, 
                    **{
                        'data__keyword': benchmark_dataset, 
                    }
                }

                ### calculate iteration ###            
                x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator = generate_dataset(dataset_config, np.random.RandomState(0))
                if x_test.shape[0] > 6000:
                    n_iter = 1
                elif x_test.shape[0] > 3000:
                    n_iter = 2
                elif x_test.shape[0] > 1000:
                    n_iter = 3
                else:
                    n_iter = 5

                print(n_iter)

                ### build dataset for each iteration ###
                for i in range(n_iter):
                    x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator = generate_dataset(dataset_config, np.random.RandomState(i))
                    raw_x_train, raw_x_val, raw_x_test, _, _, _, categorical_indicator = generate_dataset(dataset_config_rawx, np.random.RandomState(i))
                    col_cat_count = calc_col_cat_count(x_train, x_val, x_test, categorical_indicator)

                    if dataset_config['regression']:
                        target_transformer = QuantileTransformer(output_distribution='normal')
                        y_train_transform = target_transformer.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
                        y_val_transform = target_transformer.transform(y_val.reshape(-1, 1)).reshape(-1)
                        y_test_transform = target_transformer.transform(y_test.reshape(-1, 1)).reshape(-1)
                    else:
                        target_transformer = None
                        y_train_transform = None
                        y_val_transform = None
                        y_test_transform = None
                        
                    label_cat_count = calc_label_cat_count(y_train, y_val, y_test, dataset_config['regression'])

                    ### wrap data as dictionary ###
                    data_save_dir = f'{main_dir[benchmark_size]}/{benchmark_dataset}'
                    data_save_path = f'{data_save_dir}/{i}.pkl'
                    data_dict = {
                        'x_train': x_train,
                        'x_train_raw': raw_x_train,
                        'x_val': x_val,
                        'x_val_raw': raw_x_val,
                        'x_test': x_test,
                        'x_test_raw': raw_x_test,
                        'y_train': y_train,
                        'y_val': y_val,
                        'y_test': y_test,
                        'y_train_transform': y_train_transform,
                        'y_val_transform': y_val_transform,
                        'y_test_transform': y_test_transform,
                        'col_cat_count': col_cat_count,
                        'label_cat_count': label_cat_count,
                        'target_transformer': target_transformer,
                        'dataset_config': dataset_config
                    }

                    ### save data ###
                    if not os.path.exists(data_save_dir):
                        os.makedirs(data_save_dir)

                    with open(data_save_path, 'wb') as f:
                        pickle.dump(data_dict, f)            
