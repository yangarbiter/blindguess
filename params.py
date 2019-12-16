
from utils import SampleExperiments

random_seed = list(range(1))

class sample_experiments(SampleExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "sample experiment"
        grid_params = []
        grid_params.append({
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return SampleExperiments.__new__(cls, *args, **kwargs)

class compare_algos_const(StreamingExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "compare_algos"
        grid_params = []
        grid_params.append({
            'learner': [f'const-biooja-{10**i}' for i in range(1, 5)] + [f'const-oja-{10**i}' for i in range(4, 8)],
            'dataset': ['synthetic00-10000-10'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return StreamingExperiments.__new__(cls, *args, **kwargs)

class compare_algos_01(StreamingExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "compare_algos"
        grid_params = []
        grid_params.append({
            'learner': [f'biooja-{10**i}' for i in range(1, 8)] + [f'oja-{10**i}' for i in range(1, 8)],
            'dataset': ['synthetic01-10000-10'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return StreamingExperiments.__new__(cls, *args, **kwargs)

class compare_algos_const_01(StreamingExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "compare_algos"
        grid_params = []
        grid_params.append({
            'learner': [f'const-biooja-{10**i}' for i in range(1, 5)] + [f'const-oja-{10**i}' for i in range(4, 8)],
            'dataset': ['synthetic01-10000-10'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return StreamingExperiments.__new__(cls, *args, **kwargs)
