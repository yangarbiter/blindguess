import numpy as np

from autovar.base import RegisteringChoiceType, register_var, VariableClass

class DatasetVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines the dataset to use"""
    var_name = 'dataset'

    @register_var(argument=r"smallmnist", shown_name="smallmnist")
    @staticmethod
    def smallmnist(auto_var, var_value, inter_var):
        from tensorflow.keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        trnX, trny, tstX, tsty = [], [], [], []
        for i in np.unique(y_train):
            trnX.append(x_train[y_train==i][:20])
            trny.append(np.ones(20) * i)
            tstX.append(x_test[y_test==i][:20])
            tsty.append(np.ones(20) * i)

        return np.vstack(trnX), np.concatenate(trny), np.vstack(tstX), np.concatenate(tsty)

    @register_var(argument=r"mnist", shown_name="mnist")
    @staticmethod
    def mnist(auto_var, var_value, inter_var):
        from tensorflow.keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test

    @register_var(argument=r"fashion", shown_name="fashion mnist")
    @staticmethod
    def fashion(auto_var, var_value, inter_var):
        from tensorflow.keras.datasets import fashion_mnist

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test

    @register_var(argument=r"smallcifar10", shown_name="smallCifar10")
    @staticmethod
    def smallcifar10(auto_var, var_value, inter_var):
        from tensorflow.keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        trnX, trny, tstX, tsty = [], [], [], []
        for i in np.unique(y_train):
            trnX.append(x_train[y_train==i][:20])
            trny.append(np.ones(20) * i)
            tstX.append(x_test[y_test==i][:20])
            tsty.append(np.ones(20) * i)

        return np.vstack(trnX), np.concatenate(trny), np.vstack(tstX), np.concatenate(tsty)

    @register_var(argument=r"cifar10", shown_name="Cifar10")
    @staticmethod
    def cifar10(auto_var, var_value, inter_var):
        from tensorflow.keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test

    @register_var(argument=r"svhn", shown_name="SVHN")
    @staticmethod
    def svhn(auto_var, var_value, inter_var):
        #from tensorflow.keras.datasets import svhn_cropped
        from torchvision.datasets import SVHN

        trn_svhn = SVHN("./data/", split='train', download=True)
        tst_svhn = SVHN("./data/", split='test', download=True)

        x_train, y_train, x_test, y_test = [], [], [], []
        for x, y in trn_svhn:
            x_train.append(np.array(x).reshape(32, 32, 3))
            y_train.append(y)
        for x, y in tst_svhn:
            x_test.append(np.array(x).reshape(32, 32, 3))
            y_test.append(y)
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        x_test, y_test = np.asarray(x_test), np.asarray(y_test)

        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test

    @register_var(argument=r"staircase(?P<n_samples>[0-9]+)", shown_name="staircase")
    @staticmethod
    def staircase(auto_var, inter_var, n_samples):
        pass