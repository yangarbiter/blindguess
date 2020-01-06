
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
                f'strades6ce-tor-{arch}',
                f'stradesce-tor-{arch}',
                f'pstrades6ce-tor-{arch}',
                f'pstradesce-tor-{arch}',
                f'ce-tor-{arch}',
                #f'tradesce-tor-{arch}',
                f'trades10ce-tor-{arch}',
                #f'trades20ce-tor-{arch}',
                #f'ptrades6ce-tor-{arch}',
                #f'curece-tor-{arch}',
                f'cure14ce-tor-{arch}',
                #f'cure68ce-tor-{arch}',
                f'advce-tor-{arch}',
                #f'llr65ce-tor-{arch}',
                f'llrce-tor-{arch}',
                f'gr1e3ce-tor-{arch}',
                f'advkld-tor-{arch}',
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
                f'strades6ce-tor-{arch}',
                f'stradesce-tor-{arch}',
                f'pstrades6ce-tor-{arch}',
                f'pstradesce-tor-{arch}',
                f'ce-tor-{arch}',
                #f'trades10ce-tor-{arch}',
                #f'ptrades10ce-tor-{arch}',
                f'cure14ce-tor-{arch}',
                f'advce-tor-{arch}',
                f'llrce-tor-{arch}',
                f'gr1e3ce-tor-{arch}',
                f'advkld-tor-{arch}',
            ],
            'eps': [0.1],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class noisyMnistLip(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "noisymnist"
        cls.experiment_fn = 'experiment01'
        grid_params = []
        arch = "CNN002"
        grid_params.append({
            'dataset': ['noisymnist-0.15', 'noisyfashion-0.1'],
            'model': [
                f'strades6ce-tor-{arch}',
                f'stradesce-tor-{arch}',
                f'ce-tor-{arch}',
                #f'trades6ce-tor-{arch}',
                #f'trades10ce-tor-{arch}',
                #f'trades20ce-tor-{arch}',
                f'ptrades6ce-tor-{arch}',
                #f'curece-tor-{arch}',
                #f'cure14ce-tor-{arch}',
                f'advce-tor-{arch}',
                #f'llrce-tor-{arch}',
                #f'gr4ce-tor-{arch}',
                #f'kld-tor-{arch}',
                #f'advkld-tor-{arch}',
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
                f'strades6ce-tor-{arch}',
                f'stradesce-tor-{arch}',
                f'pstrades6ce-tor-{arch}',
                f'pstradesce-tor-{arch}',
                #f'liplkld-tor-{arch}',
                #f'liplce-tor-{arch}',
                f'ce-tor-{arch}',
                #f'tradesce-tor-{arch}',
                #f'trades6ce-tor-{arch}',
                #f'trades10ce-tor-{arch}',
                #f'trades20ce-tor-{arch}',
                #f'ptrades6ce-tor-{arch}',
                #f'ptrades20ce-tor-{arch}',
                #f'curece-tor-{arch}',
                f'cure14ce-tor-{arch}',
                #f'cure68ce-tor-{arch}',
                f'advce-tor-{arch}',
                #f'llr65ce-tor-{arch}',
                f'llrce-tor-{arch}',
                f'gr4ce-tor-{arch}',
                f'gr1e3ce-tor-{arch}',
                f'kld-tor-{arch}',
                f'advkld-tor-{arch}',
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
                f'strades6ce-tor-{arch}',
                f'stradesce-tor-{arch}',
                f'pstrades6ce-tor-{arch}',
                f'pstradesce-tor-{arch}',
                #f'liplkld-tor-{arch}',
                #f'liplce-tor-{arch}',
                f'ce-tor-{arch}',
                #f'trades10ce-tor-{arch}',
                #f'trades20ce-tor-{arch}',
                #f'ptrades6ce-tor-{arch}',
                #f'curece-tor-{arch}',
                f'cure14ce-tor-{arch}',
                f'advce-tor-{arch}',
                #f'advce-tor-{arch}-lrem3',
                f'llrce-tor-{arch}',
                f'gr4ce-tor-{arch}',
                f'gr1e3ce-tor-{arch}',
                f'kld-tor-{arch}',
                f'advkld-tor-{arch}',
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
                f'strades6ce-tor-{arch}',
                f'stradesce-tor-{arch}',
                f'pstrades6ce-tor-{arch}',
                f'pstradesce-tor-{arch}',
                f'advce-tor-{arch}',
                #f'cure14ce-tor-{arch}',
            ],
            'eps': [0.1],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
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
                'strades6ce-tor-WRN_40_10',
                'pstrades6ce-tor-WRN_40_10',
                'trades6ce-tor-WRN_40_10',
                'curece-tor-WRN_40_10',
                'curece14-tor-WRN_40_10',
                'advce-tor-WRN_40_10-lrem2',
                'llrce-tor-WRN_40_10',
                'advkld-tor-WRN_40_10',
                'gr1e3ce-tor-WRN_40_10',
            ],
            'eps': [0.031],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class svhnOtherLips(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "svhn"
        cls.experiment_fn = 'experiment03'
        grid_params = []
        grid_params.append({
            'dataset': ['svhn'],
            'model': [
                'ce-tor-WRN_40_10',
                'strades6ce-tor-WRN_40_10',
                'strades10ce-tor-WRN_40_10',
                'pstrades6ce-tor-WRN_40_10',
                'curece-tor-WRN_40_10',
                'cure14ce-tor-WRN_40_10',
                'advce-tor-WRN_40_10-lrem2',
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
                'strades6ce-tor-WRN_40_10',
                'trades6ce-tor-WRN_40_10',
                'cure14ce-tor-WRN_40_10',
                'advce-tor-WRN_40_10',
                'advce-tor-WRN_40_10-lrem2',
                'llrce-tor-WRN_40_10',
                'gr4ce-tor-WRN_40_10',
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

class cifarOtherLips(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "cifar"
        cls.experiment_fn = 'experiment03'
        grid_params = []
        grid_params.append({
            'dataset': ['cifar10'],
            'model': [
                'ce-tor-WRN_40_10',
                'trades6ce-tor-WRN_40_10',
                'cure14ce-tor-WRN_40_10',
                'advce-tor-WRN_40_10',
                'advce-tor-WRN_40_10-lrem2',
                'llrce-tor-WRN_40_10',
            ],
            'eps': [0.031],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)
