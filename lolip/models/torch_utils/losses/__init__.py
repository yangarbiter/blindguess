
from .trades import trades_loss
from ....attacks.torch.projected_gradient_descent import projected_gradient_descent

def get_outputs_loss(model, optimizer, base_loss_fn, x, y, reduction, **kwargs):

    norm = kwargs['norm']
    eps = kwargs['eps']
    device = kwargs['device']
    clip_img = kwargs['clip_img']
    loss_name = kwargs['loss_name']

    if 'trades' in loss_name:
        #re.search('trades', loss_name)
        if 'trades10' in loss_name:
            beta = 10.0
        elif 'trades20' in loss_name:
            beta = 20.0
        elif 'trades16' in loss_name:
            beta = 16.0
        elif 'trades6' in loss_name:
            beta = 6.0
        elif 'trades.5' in loss_name:
            beta = 0.5
        elif 'trades.1' in loss_name:
            beta = 0.1
        else:
            beta = 1.0

        if 'K20' in loss_name:
            steps = 20
        else:
            steps = 10

        #print(f"TRADES version: {version}")
        if clip_img:
            clip_min, clip_max = 0, 1

        if 'truntrades' in loss_name:
            pass
        else:
            outputs, loss = trades_loss(
                model, base_loss_fn, x, y,
                norm=norm, optimizer=optimizer, clip_min=clip_min, clip_max=clip_max,
                step_size=eps*2/steps, epsilon=eps, perturb_steps=steps, beta=beta,
                device=device
            )
    elif 'lipz' in loss_name:
        pass
    elif 'rbfw' in loss_name:
        optimizer.zero_grad()
        outputs, loss = rbfw_loss(
            model, base_loss_fn, x, y, norm=norm, gamma=model.gamma_var,
        )
    else:
        if 'adv' in loss_name:
            clip_min, clip_max = None, None
            if clip_img:
                clip_min, clip_max = 0, 1
            x = projected_gradient_descent(model, x, y=y, clip_min=clip_min,
                clip_max=clip_max, eps_iter=eps/5, eps=eps, norm=norm, nb_iter=10)

        optimizer.zero_grad()
        outputs = model(x)
        loss = base_loss_fn(outputs, y)

    return outputs, loss
