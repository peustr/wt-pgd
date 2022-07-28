import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_EOT_SAMPLES = 16
DEFAULT_WT_SAMPLES = 16
DEFAULT_WT_STD = 0.05


def wtpgd(
    model,
    data,
    target,
    epsilon=8./255.,
    num_iter=10,
    step_size=0.1,
    random_start=True,
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
        random_start: Add uniform noise in [-epsilon, epsilon] before the
            first PGD step.
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
    step_size = step_size * epsilon
    if random_start:
        with torch.no_grad():
            perturbed_data.data += torch.empty_like(
                perturbed_data.data).uniform_(-epsilon, epsilon)
            perturbed_data.data.clamp_(data_min, data_max)
    for pgd_iter in range(num_iter):
        wt_grad_samples = []
        for wt_iter in range(num_wt_samples):
            if wt_iter != 0:
                u = perturbed_data + torch.randn(
                    perturbed_data.shape,
                    dtype=perturbed_data.dtype,
                    device=perturbed_data.device
                ) * wt_std
            else:
                u = perturbed_data
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
            perturbed_data.data.clamp_(data_min, data_max)
    perturbed_data.requires_grad = False
    return perturbed_data


def fixed_point_wtpgd(
    model,
    data,
    target,
    num_eot_samples=DEFAULT_EOT_SAMPLES,
    num_wt_samples=DEFAULT_WT_SAMPLES,
    wt_std=DEFAULT_WT_STD,
):
    """
    Params:
        model: The model under attack.
        data: A batch of images.
        target: Class labels.
        num_eot_samples: Number of EoT samples to deal with stochastic gradients.
            Only meaningful with stochastic defences, use 1 otherwise.
        num_wt_samples: Number of images to sample for the Weierstrass transform.
        wt_std: The standard deviation of the Weierstrass transform sampling method.
    """
    model.eval()
    perturbed_data = data.clone()
    wt_samples = []
    for wt_iter in range(num_wt_samples):
        if wt_iter != 0:
            delta = torch.randn(
                perturbed_data.shape,
                dtype=perturbed_data.dtype,
                device=perturbed_data.device
            ) * wt_std
        else:
            delta = 0.
        wt_samples.append(perturbed_data + delta)
        wt_samples.append(perturbed_data - delta)
    wt_samples = torch.cat(wt_samples, dim=0)
    eot_loss_samples = []
    for eot_iter in range(num_eot_samples):
        output = model(wt_samples)
        loss = F.cross_entropy(output, target.repeat(2 * num_wt_samples)).item()
        eot_loss_samples.append(loss)
    return np.mean(eot_loss_samples)


def wtzoo(
    model,
    data,
    target,
    epsilon=8./255.,
    num_iter=10,
    step_size=0.1,
    c=0.0001,
    p=0.05,
    kappa=0.,
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
    def hinge_loss(model, data, target):
        """ Eq. 5 in original ZOO paper, with kappa=0. """
        posterior = model(data).softmax(-1)
        return (
            posterior.softmax(-1).log().gather(1, target.view(-1, 1)) -
            posterior.softmax(-1).log().max(-1).values[:, None]
        )

    model.eval()
    perturbed_data = data.clone()
    perturbed_data.requires_grad = True
    data_min = data - epsilon
    data_max = data + epsilon
    step_size = step_size * epsilon
    for zoo_iter in range(num_iter):
        e = torch.rand_like(perturbed_data)
        e[e >= 1 - p] = 1.
        e[e < 1 - p] = 0.
        wt_delta_samples = []
        for wt_iter in range(num_wt_samples):
            if wt_iter != 0:
                u = perturbed_data + torch.randn(
                    perturbed_data.shape,
                    dtype=perturbed_data.dtype,
                    device=perturbed_data.device
                ) * wt_std
            else:
                u = perturbed_data
            eot_delta_samples = []
            for eot_iter in range(num_eot_samples):
                f1 = hinge_loss(model, u + c * e, target, kappa=kappa)
                f2 = hinge_loss(model, u, target, kappa=kappa)
                f3 = hinge_loss(model, u - c * e, target, kappa=kappa)
                grad = (f1 - f3) / 2 * c  # Eq. 6 in original ZOO paper.
                hess = (f1 - 2 * f2 + f3) / c ** 2  # Eq. 7 in original ZOO paper.
                assert grad.shape == hess.shape
                delta = torch.zeros_like(grad)
                delta[hess <= 0.] = -step_size * grad[hess <= 0.]
                delta[hess > 0] = -step_size * (grad[hess > 0.] / hess[hess > 0.])
                eot_delta_samples.append(delta.clone())
            wt_delta_samples.append(torch.stack(eot_delta_samples).mean(0))
        delta_star = torch.stack(wt_delta_samples).mean(0)
        perturbed_data.data += e * delta_star[:, None, None]
        perturbed_data.data.clamp_(data_min, data_max)
    return perturbed_data
