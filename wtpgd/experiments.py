import numpy as np
import torch
import torch.nn.functional as F


def get_original_loss_landscape(
    model,
    data,
    target,
    epsilon=8./255.,
    num_epsilons=100,
    num_eot_samples=1,
    device='cpu',
):
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
    g2 = g1.clone()
    # compute an orthogonal axis
    g2[-1] = -(g1[:-1] @ g1[:-1]) / g1[-1]
    g2 = g2 / g2.norm()
    g1 = g1.to(device)
    g2 = g2.to(device)
    x, y, z = [], [], []
    u_ = data.reshape(1 * 3 * 32 * 32)
    epsilons = np.linspace(-epsilon, epsilon, num_epsilons)
    for e1 in epsilons:
        for e2 in epsilons:
            u_prime = (u_ + e1 * torch.sign(g1) + e2 * torch.sign(g2)).reshape(1, 3, 32, 32)
            logits = model(u_prime)
            loss = F.cross_entropy(logits, target)
            x.append(e1)
            y.append(e2)
            z.append(loss.item())
    return x, y, z, g1, g2
