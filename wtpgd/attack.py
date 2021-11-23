import torch
import torch.nn.functional as F


def wtpgd(
    model,
    data,
    target,
    epsilon=8./255.,
    k=10,
    a=0.01,
    d_min=0.,
    d_max=1.,
    num_eot_samples=16,
    num_wt_samples=16,
    wt_std=0.045,
):
    model.eval()
    perturbed_data = data.clone()
    perturbed_data.requires_grad = True
    data_min = data - epsilon
    data_max = data + epsilon
    data_min.clamp_(d_min, d_max)
    data_max.clamp_(d_min, d_max)
    for pgd_iter in range(k):
        wt_grad_samples = []
        for wt_iter in range(num_wt_samples):
            u = perturbed_data + torch.randn(
                perturbed_data.shape,
                dtype=perturbed_data.dtype,
                device=perturbed_data.device
            ) * wt_std
            eot_grad_samples = []
            for eot_iter in range(num_eot_samples):
                if perturbed_data.grad is not None:
                    perturbed_data.grad.data.zero_()
                output = model(u)
                loss = F.cross_entropy(output, target)
                loss.backward()
                eot_grad_samples.append(perturbed_data.grad.data.clone())
            wt_grad_samples.append(torch.stack(eot_grad_samples).mean(0))
        grad_sign = torch.stack(wt_grad_samples).mean(0).sign()
        with torch.no_grad():
            perturbed_data.data += a * grad_sign
            perturbed_data.data = torch.max(torch.min(perturbed_data, data_max), data_min)
    perturbed_data.requires_grad = False
    return perturbed_data
