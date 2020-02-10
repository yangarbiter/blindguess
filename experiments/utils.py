import numpy as np
import tensorflow as tf
import torch

def set_random_seed(auto_var):
    random_seed = auto_var.get_var("random_seed")

    torch.manual_seed(random_seed)
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    random_state = np.random.RandomState(auto_var.get_var("random_seed"))
    auto_var.set_intermidiate_variable("random_state", random_state)

    return random_state

def load_model(auto_var, trnX, trny, tstX, tsty, n_channels):
    model = auto_var.get_var("model", trnX=trnX, trny=trny, n_channels=n_channels)
    if tstX is not None and tsty is not None:
        model.tst_ds = (tstX, tsty)
    model_path = get_file_name(auto_var).split("-")
    model_path[0] = 'pgd'
    model_path = '-'.join(model_path)
    if os.path.exists(os.path.join('./models', model_path + '-ep%04d.pt') % model.epochs):
        model_path = os.path.join('./models', model_path + '-ep%04d.pt') % model.epochs
    else:
        model_path = os.path.join('./models', model_path + '.pt')

    model.load(model_path)
    model.model.cuda()
    return model_path, model
