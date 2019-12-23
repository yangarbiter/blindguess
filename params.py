
from utils import ExpExperiments

random_seed = list(range(1))

class mnistLip(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "mnist"
        cls.experiment_fn = 'experiment01'
        grid_params = []
        grid_params.append({
            'dataset': ['mnist', 'fashion'],
            'model': ['ce-tor-CNN001', 'tradesce-tor-CNN001', 'advce-tor-CNN001'],
            'eps': [1.0],
            'norm': ['2'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class cifarLip(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "cifar"
        cls.experiment_fn = 'experiment01'
        grid_params = []
        grid_params.append({
            'dataset': ['cifar10'],
            'model': ['ce-tor-WideResNet', 'tradesce-tor-WideResNet', 'advce-tor-WideResNet'],
            'eps': [10.0],
            'norm': ['2'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)
