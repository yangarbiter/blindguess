
from .trades import trades_loss
from .l1lip import l1lip_loss
from ....attacks.torch.projected_gradient_descent import projected_gradient_descent
from ....attacks.torch.spatials.grid import grid_spatial_attack
from ....attacks.torch.spatials.fft_method import first_order_attack_spatial_fft
from ....attacks.torch.fft_epsilon import first_order_attack_fft

def get_outputs_loss(model, optimizer, base_loss_fn, x, y, reduction, **kwargs):

    norm = kwargs['norm']
    eps = kwargs['eps']
    device = kwargs['device']
    clip_img = kwargs['clip_img']
    loss_name = kwargs['loss_name']

    if 'trades' in loss_name:
        if 'trades10' in loss_name:
            beta = 10.0
        elif 'trades20' in loss_name:
            beta = 20.0
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
        elif 'htrades' in loss_name:
            version = "hard"
        elif 'wtrades' in loss_name:
            version = "weighted"
        else:
            version = None
        outputs, loss = trades_loss(
            model, base_loss_fn, x, y,
            norm=norm, optimizer=optimizer, clip_min=clip_min, clip_max=clip_max,
            step_size=eps*2/steps, epsilon=eps, perturb_steps=steps, beta=beta,
            device=device, version=version
        )
        loss = loss.mean()
    elif 'l1lip' in loss_name:
        if 'l1lip10' in loss_name:
            beta = 10.0
        elif 'l1lip20' in loss_name:
            beta = 20.0
        elif 'l1lip6' in loss_name:
            beta = 6.0
        elif 'l1lip.5' in loss_name:
            beta = 0.5
        elif 'l1lip.1' in loss_name:
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

        if 'trunl1lip' in loss_name:
            pass
        elif 'wtrades' in loss_name:
            version = "weighted"
        else:
            version = None
        outputs, loss = l1lip_loss(
            model, base_loss_fn, x, y,
            norm=norm, optimizer=optimizer, clip_min=clip_min, clip_max=clip_max,
            step_size=eps*2/steps, epsilon=eps, perturb_steps=steps, beta=beta,
            device=device, version=version
        )
        loss = loss.mean()
    elif 'rbfw' in loss_name:
        optimizer.zero_grad()
        outputs, loss = rbfw_loss(
            model, base_loss_fn, x, y, norm=norm, gamma=model.gamma_var,
        )
    else:
        if 'spgridadv' in loss_name:
            # spatial shift
            _, x = grid_spatial_attack(x, y, model,
                    base_loss_fn, 0, 0.2, 0, device)
        elif 'spfftadv' in loss_name:
            # fft shift
            _, x = first_order_attack_spatial_fft(x, y, model,
                    base_loss_fn, 10, 0.1, 0, 0.2, 0, device)
        elif 'fftadv' in loss_name:
            # fft shift
            _, x = first_order_attack_fft(x, y, model, base_loss_fn,
                    eps=eps, perturb_iters=10, step_size=eps/10, device=device)
        elif 'adv' in loss_name:
            clip_min, clip_max = None, None
            if clip_img:
                clip_min, clip_max = 0, 1
            x = projected_gradient_descent(model, x, y=y, clip_min=clip_min,
                clip_max=clip_max, eps_iter=eps/5, eps=eps, norm=norm, nb_iter=10)

        optimizer.zero_grad()
        outputs = model(x)
        loss = base_loss_fn(outputs, y)
        loss = loss.mean()

    return outputs, loss
