import numpy as np
import torch
import torch.nn.functional as F

from wtpgd.attacks import DEFAULT_EOT_SAMPLES, DEFAULT_WT_SAMPLES, DEFAULT_WT_STD, wtpgd


def get_loss_landscape(
    model,
    data,
    target,
    epsilon=8./255.,
    num_epsilons=100,
    num_eot_samples=DEFAULT_EOT_SAMPLES,
    use_wtpgd=False,
    wtpgd_args=None,
    gradient_axes=None,
):
    """
    Params:
        model: The model that implements the adversarial defence.
        data: A single data point (batch_size=1) of shape 1x3x32x32 (e.g., a CIFAR image).
        target: The class label of the image.
        epsilon: The epsilon-ball around the initial image.
        num_epsilons: Number of evenly-spaced images to sample in the epsilon-ball.
        num_eot_samples: EoT iterations for getting a more reliable axis gradient. Use 1 for
            non-stochastic defences.
        use_wtpgd: Get the loss landscape when under attack by WTPGD.
        wtpgd_args: Arguments for WTPGD, if use_wtpgd is True. Pass them in a dictionary
            format like: {num_eot_samples=?, num_wt_samples=?, wt_std=?}.
        gradient_axes: A tuple of (g1, g2), where g1 and g2 are orthogonal, to project the
            loss. Seeding the axes this way is helpful when we want to visualise the loss
            surface of the target model on the gradient axes of the original model.
    """
    if gradient_axes is None:
        data.requires_grad = True
        eot_grad_samples = []
        for eot_iter in range(num_eot_samples):
            if data.grad is not None:
                data.grad.data.zero_()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            eot_grad_samples.append(data.grad.data.clone())
        data.requires_grad = False
        eot_grad = torch.stack(eot_grad_samples).mean(0)
        g1 = eot_grad.reshape(1 * 3 * 32 * 32)
        g1 = g1 / g1.norm()
        g1 = g1.to(data.device)
        # compute an orthogonal axis
        g2 = g1.clone()
        g2[-1] = -(g1[:-1] @ g1[:-1]) / g1[-1]
        g2 = g2 / g2.norm()
        g2 = g2.to(data.device)
    else:
        g1, g2 = gradient_axes
    x, y, z = [], [], []
    epsilons = np.linspace(-epsilon, epsilon, num_epsilons)
    for e1 in epsilons:
        for e2 in epsilons:
            if use_wtpgd:
                u_prime = wtpgd(
                    model, data, target, epsilon=epsilon,
                    num_eot_samples=wtpgd_args.get('num_eot_samples', DEFAULT_EOT_SAMPLES),
                    num_wt_samples=wtpgd_args.get('num_wt_samples', DEFAULT_WT_SAMPLES),
                    wt_std=wtpgd_args.get('wt_std', DEFAULT_WT_STD),
                )
            else:
                u_ = data.reshape(1 * 3 * 32 * 32)
                u_prime = (u_ + e1 * torch.sign(g1) + e2 * torch.sign(g2)).reshape(1, 3, 32, 32)
            logits = model(u_prime)
            loss = F.cross_entropy(logits, target)
            x.append(e1)
            y.append(e2)
            z.append(loss.item())
    return x, y, z, g1, g2
