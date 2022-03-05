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
    num_iter=10,
    step_size=0.01,
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
        target: Class labels.
        epsilon: Attack strength.
        num_iter: Number of iterations.
        step_size: Learning rate.
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
    for pgd_iter in range(num_iter):
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
            perturbed_data.data += step_size * grad_sign
            perturbed_data.data = torch.max(torch.min(perturbed_data, data_max), data_min)
    perturbed_data.requires_grad = False
    return perturbed_data


def wtzoo(
    model,
    data,
    target,
    epsilon=8./255.,
    num_iter=10,
    step_size=0.01,
    c=0.0001,
    p=0.05,
    kappa=0.,
    d_min=0.,
    d_max=1.,
    num_eot_samples=DEFAULT_EOT_SAMPLES,
    num_wt_samples=DEFAULT_WT_SAMPLES,
    wt_std=DEFAULT_WT_STD,
):
    """
    Params:
        model: Model under attack.
        data: Batch of images.
        target: Class labels.
        epsilon: Attack strength.
        num_iter: Number of iterations.
        step_size: Learning rate.
        c: Small constant used for gradient approximation.
        p: Percentage of pixels to perturb in each iteration.
        kappa: Confidence bound for the hinge loss.
        d_min, d_max: Upper and lower pixel values for images (usually normalised
            in [0, 1]).
        num_eot_samples: Number of EoT samples to deal with stochastic gradients.
            Only meaningful with stochastic defences, use 1 otherwise.
        num_wt_samples: Number of images to sample for the Weierstrass transform.
        wt_std: The standard deviation of the Weierstrass transform sampling method.
    """
    def hinge_loss(model, data, target, kappa=0.):
        posterior = model(data).softmax(-1)
        return (
            posterior.softmax(-1).log() -
            posterior.softmax(-1).log().gather(1, target.view(-1, 1)) -
            kappa
        ).max(-1).values

    model.eval()
    perturbed_data = data.clone()
    data_min = data - epsilon
    data_max = data + epsilon
    data_min.clamp_(d_min, d_max)
    data_max.clamp_(d_min, d_max)
    for zoo_iter in range(num_iter):
        e = torch.rand_like(perturbed_data)
        e[e >= 1 - p] = 1.
        e[e < 1 - p] = 0.
        wt_delta_samples = []
        for wt_iter in range(num_wt_samples):
            u = perturbed_data + torch.randn(
                perturbed_data.shape,
                dtype=perturbed_data.dtype,
                device=perturbed_data.device
            ) * wt_std
            eot_delta_samples = []
            for eot_iter in range(num_eot_samples):
                l1 = hinge_loss(model, u + c * e, target, kappa=kappa)
                l2 = hinge_loss(model, u, target, kappa=kappa)
                l3 = hinge_loss(model, u - c * e, target, kappa=kappa)
                g = (l1 - l3) / 2 * c
                h = (l1 - 2 * l2 + l3) / c ** 2
                assert g.shape == h.shape
                delta = torch.zeros_like(g)
                delta[h <= 0.] = -step_size * g[h <= 0.]
                delta[h > 0] = -step_size * (g[h > 0.] / h[h > 0.])
                eot_delta_samples.append(delta.clone())
            wt_delta_samples.append(torch.stack(eot_delta_samples).mean(0))
        delta_star = torch.stack(wt_delta_samples).mean(0)
        perturbed_data.data += e * delta_star[:, None, None]
        perturbed_data.data = torch.max(torch.min(perturbed_data, data_max), data_min)
    return perturbed_data
