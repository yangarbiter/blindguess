
from utils import ExpExperiments

random_seed = list(range(1))

class mnistOtherLips(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "mnist"
        cls.experiment_fn = 'experiment03'
        grid_params = []
        arch = "CNN001"
        grid_params.append({
            'dataset': ['mnist'],
            'model': [
                f'strades6ce-tor-{arch}',
                f'stradesce-tor-{arch}',
                f'ce-tor-{arch}',
                f'tulipce-tor-{arch}',
                #f'cure14ce-tor-{arch}',
                f'advce-tor-{arch}',
                f'llrce-tor-{arch}',
            ],
            'eps': [0.1],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        arch = "CNN002"
        grid_params.append({
            'dataset': ['mnist'],
            'model': [
                f'strades6ce-tor-{arch}',
                f'stradesce-tor-{arch}',
                f'ce-tor-{arch}',
                f'tulipce-tor-{arch}',
                #f'cure14ce-tor-{arch}',
                f'advce-tor-{arch}',
                f'llrce-tor-{arch}',
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
            'dataset': ['mnist'],
            'model': [
                f'strades6ce-tor-{arch}',
                f'stradesce-tor-{arch}',
                #f'liplkld-tor-{arch}',
                #f'liplce-tor-{arch}',
                f'tulipce-tor-{arch}',
                f'ce-tor-{arch}',
                #f'cure14ce-tor-{arch}',
                f'advce-tor-{arch}',
                f'llrce-tor-{arch}',
            ],
            'eps': [0.1],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        arch = "CNN002"
        grid_params.append({
            'dataset': ['mnist'],
            'model': [
                f'strades6ce-tor-{arch}',
                f'stradesce-tor-{arch}',
                f'tulipce-tor-{arch}',
                f'ce-tor-{arch}',
                #f'cure14ce-tor-{arch}',
                f'advce-tor-{arch}',
                f'llrce-tor-{arch}',
            ],
            'eps': [0.1],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class mnistFixLips(mnistOtherLips):
    def __new__(cls, *args, **kwargs):
        return mnistOtherLips.__new__(cls, *args, **kwargs)
    def __init__(self):
        self.experiment_fn = 'experiment02'

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
                'strades10ce-tor-WRN_40_10',
                #'curece-tor-WRN_40_10',
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
                'stradesce-tor-WRN_40_10',
                'pstrades6ce-tor-WRN_40_10',
                #'cure14ce-tor-WRN_40_10',
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
                'tulipce-tor-WRN_40_10',
                'strades6ce-tor-WRN_40_10',
                'stradesce-tor-WRN_40_10',
                #'cure14ce-tor-WRN_40_10',
                'advce-tor-WRN_40_10',
                'llrce-tor-WRN_40_10',
            ],
            'eps': [0.031],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'dataset': ['cifar10'],
            'model': [
                'aug01-ce-tor-WRN_40_10',
                'aug01-tulipce-tor-WRN_40_10',
                'aug01-strades6ce-tor-WRN_40_10',
                'aug01-stradesce-tor-WRN_40_10',
                'aug01-advce-tor-WRN_40_10-lrem2',
                #'aug01-scure14ce-tor-WRN_40_10-lrem4',
                'aug01-llrce-tor-WRN_40_10',
                'aug01-sllrce-tor-WRN_40_10',
            ],
            'eps': [0.031],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
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
                'tulipce-tor-WRN_40_10',
                'stradesce-tor-WRN_40_10',
                'strades6ce-tor-WRN_40_10',
                'advce-tor-WRN_40_10',
                'llrce-tor-WRN_40_10',
            ],
            'eps': [0.031],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'dataset': ['cifar10'],
            'model': [
                'aug01-ce-tor-WRN_40_10',
                'aug01-tulipce-tor-WRN_40_10',
                'aug01-strades6ce-tor-WRN_40_10',
                'aug01-stradesce-tor-WRN_40_10',
                'aug01-advce-tor-WRN_40_10-lrem2',
                'aug01-llrce-tor-WRN_40_10',
            ],
            'eps': [0.031],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class cifarFixLips(cifarOtherLips):
    def __new__(cls, *args, **kwargs):
        return cifarOtherLips.__new__(cls, *args, **kwargs)
    def __init__(self):
        self.experiment_fn = 'experiment02'

class tinyLip(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "tiny ImageNet"
        cls.experiment_fn = 'experiment01'
        grid_params = []
        arch = "ResNet152"
        grid_params.append({
            'dataset': ['tinyimgnet'],
            'model': [
                f'aug02-ce-tor-{arch}-bs256',
                f'aug02-strades6ce-tor-{arch}-bs256',
                f'aug02-advce-tor-{arch}-bs256',
                f'aug02-llrce-tor-{arch}',
            ],
            'eps': [0.031],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class resImgOtherLips(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "Restricted ImageNet"
        cls.experiment_fn = 'restrictedImgnet2'
        grid_params = []
        arch = "ResNet152"
        grid_params.append({
            'dataset': ['resImgnet112v3'],
            'model': [
                f'ce-tor-{arch}-adambs128',
                f'trades6ce-tor-{arch}-bs128',
                f'advce-tor-{arch}-adambs128',
                f'sllr36ce-tor-{arch}-adambs128',
                f'tulipce-tor-{arch}-adambs128',
            ],
            'eps': [0.005],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        arch = "ResNet50"
        grid_params.append({
            'dataset': ['resImgnet112v3'],
            'model': [
                f'ce-tor-{arch}-adambs128',
                f'stradesce-tor-{arch}-adambs128',
                f'strades6ce-tor-{arch}-adambs128',
                f'advce-tor-{arch}-adambs128',
                f'sllr36ce-tor-{arch}-adambs128',
                f'tulipce-tor-{arch}-adambs128',
            ],
            'eps': [0.005],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class resImgFixLips(resImgOtherLips):
    def __new__(cls, *args, **kwargs):
        return resImgOtherLips.__new__(cls, *args, **kwargs)
    def __init__(self):
        self.experiment_fn = 'restrictedImgnet3'
