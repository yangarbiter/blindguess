
from utils import ExpExperiments

random_seed = list(range(1))

class mnistOtherLips(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "mnist"
        cls.experiment_fn = 'experiment03'
        grid_params = []
        arch = "CNN001"
        grid_params.append({
            'dataset': ['mnist', 'fashion'],
            'model': [
                f'ce-tor-{arch}',
                f'tradesce-tor-{arch}',
                f'trades10ce-tor-{arch}',
                f'trades20ce-tor-{arch}',
                f'ptrades10ce-tor-{arch}',
                f'curece-tor-{arch}',
                f'cure68ce-tor-{arch}',
                f'advce-tor-{arch}',
                f'llr65ce-tor-{arch}',
                f'llrce-tor-{arch}',
                f'gr4ce-tor-{arch}',
            ],
            'eps': [0.1],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        arch = "CNN002"
        grid_params.append({
            'dataset': ['mnist', 'fashion'],
            'model': [
                f'ce-tor-{arch}',
                f'trades10ce-tor-{arch}',
                f'curece-tor-{arch}',
                f'advce-tor-{arch}',
                f'llrce-tor-{arch}',
                f'gr4ce-tor-{arch}',
            ],
            'eps': [0.1],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)


class mnistLip(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "mnist"
        cls.experiment_fn = 'experiment01'
        grid_params = []
        arch = "CNN001"
        grid_params.append({
            'dataset': ['mnist', 'fashion'],
            'model': [
                f'ce-tor-{arch}',
                f'tradesce-tor-{arch}',
                f'trades6ce-tor-{arch}',
                f'trades10ce-tor-{arch}',
                f'trades20ce-tor-{arch}',
                f'ptrades10ce-tor-{arch}',
                f'curece-tor-{arch}',
                f'cure68ce-tor-{arch}',
                f'advce-tor-{arch}',
                f'llr65ce-tor-{arch}',
                f'llrce-tor-{arch}',
                f'gr4ce-tor-{arch}',
            ],
            'eps': [0.1],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        arch = "CNN002"
        grid_params.append({
            'dataset': ['mnist', 'fashion'],
            'model': [
                f'ce-tor-{arch}',
                f'trades10ce-tor-{arch}',
                f'curece-tor-{arch}',
                f'advce-tor-{arch}',
                f'advce-tor-{arch}-lrem3',
                f'llrce-tor-{arch}',
                f'gr4ce-tor-{arch}',
            ],
            'eps': [0.1],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        arch = "ResNet50"
        grid_params.append({
            'dataset': ['fashion'],
            'model': [
                f'ce-tor-{arch}',
                f'trades10ce-tor-{arch}',
                f'curece-tor-{arch}',
                f'advce-tor-{arch}',
                f'llr65ce-tor-{arch}',
                f'gr4ce-tor-{arch}',
            ],
            'eps': [0.1],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        #grid_params.append({
        #    'dataset': ['mnist', 'fashion'],
        #    'model': ['ce-tor-CNN001', 'tradesce-tor-CNN001',
        #        'advce-tor-CNN001', 'llrce-tor-CNN001'],
        #    'eps': [1.0],
        #    'norm': ['2'],
        #    'attack': ['pgd'],
        #    'random_seed': random_seed,
        #})
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class svhnLip(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "svhn"
        cls.experiment_fn = 'experiment01'
        grid_params = []
        grid_params.append({
            'dataset': ['svhn'],
            'model': [
                'ce-tor-WRN_40_10',
                'trades6ce-tor-WRN_40_10',
                'curece-tor-WRN_40_10',
                'advce-tor-WRN_40_10',
                'llrce-tor-WRN_40_10',
            ],
            'eps': [0.031],
            'norm': ['inf'],
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
            'model': [
                'ce-tor-WRN_40_10',
                'trades6ce-tor-WRN_40_10',
                'curece-tor-WRN_40_10',
                'advce-tor-WRN_40_10',
                'llrce-tor-WRN_40_10',
            ],
            'eps': [0.031],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        #grid_params.append({
        #    'dataset': ['cifar10'],
        #    'model': [
        #        'ce-tor-WideResNet', 'tradesce-tor-WideResNet', #'advce-tor-WideResNet'
        #        'ce-tor-WRN_40_10', 'tradesce-tor-WRN_40_10', 'advce-tor-WRN_40_10'
        #    ],
        #    'eps': [5.0],
        #    'norm': ['2'],
        #    'attack': ['pgd'],
        #    'random_seed': random_seed,
        #})
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)
