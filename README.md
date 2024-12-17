# :dolphin: DOFEN: Deep Oblivious Forest Ensemble :dolphin:
This is the official implementation of DOFEN, a novel tree-inspired deep tabular neural network, accepted by NeurIPS 2024 ([openreview link](https://openreview.net/forum?id=umukvCdGI6))

### Installation and build benchmark dataset
```
### we use python 3.8 ###
pip install requirements.txt
source build_tabular_benchmark_data.sh
```

### How to use DOFEN
The `DOFEN_on_tabular_benchmark.ipynb` notebook provides more detailed usage and setting of DOFEN on tabular benchmark, here we provide a quick simple view:
```python
from model import DOFENTrainer
from dofen_default_config import dofen_config, train_config, eval_config

# prepare your training data
tr_x = ...
tr_y = ...

# provide dataset specific information
dofen_config['column_category_count'] = ... # list of int, number of categories for each column, set the value to -1 for numerical columns
dofen_config['n_class'] = ... # int, number of class of a dataset, please set to 2 for binary tasks, set to 'number of class' for multiclass tasks , and set to -1 for regression tasks 

# model initialize, for detail usage and descriptions of these three configs, please see docstring of DOFENTrainer
dofen_trainer = DOFENTrainer(dofen_config, train_config, eval_config)
dofen_trainer.init()

# fit dofen on training data
dofen_trainer.fit(tr_x, tr_y)

# prepare your testing data and evaluate
te_x = ...
te_y = ...
dofen_trainer.evaluate(te_x, te_y)
```