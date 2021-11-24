import torch
import torch.nn.functional as F


DEFAULT_EOT_SAMPLES = 16
DEFAULT_WT_SAMPLES = 16
DEFAULT_WT_STD = 0.045


def wtpgd(
    model,
    data,
    target,
    epsilon=8./255.,
    k=10,
    a=0.01,
    d_min=0.,
    d_max=1.,
    num_eot_samples=DEFAULT_EOT_SAMPLES,
    num_wt_samples=DEFAULT_WT_SAMPLES,
    wt_std=DEFAULT_WT_STD,
):
    """
    Params:
        model: The model under attack.
        data: A batch of images.
        target: The class labels of the images.
        epsilon: The PGD attack strength.
        k: Number of PGD iterations.
        a: PGD step size (learning rate).
        d_min, d_max: Upper and lower pixel values for images (usually normalised
            in [0, 1]).
        num_eot_samples: Number of EoT samples to deal with stochastic gradients.
            Only meaningful with stochastic defences, use 1 otherwise.
        num_wt_samples: Number of images to sample for the Weierstrass transform.
        wt_std: The standard deviation of the Weierstrass transform sampling method.
    """
    model.eval()
    perturbed_data = data.clone()
    perturbed_data.requires_grad = True
    data_min = data - epsilon
    data_max = data + epsilon
    data_min.clamp_(d_min, d_max)
    data_max.clamp_(d_min, d_max)
    with torch.no_grad():
        perturbed_data.data = data + torch.empty(
            perturbed_data.shape,
            dtype=perturbed_data.dtype,
            device=perturbed_data.device
        ).uniform_(-epsilon, epsilon)
        perturbed_data.data.clamp_(d_min, d_max)
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
