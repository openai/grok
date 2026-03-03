import logging
import torch
import numpy as np

import scipy.optimize


def get_loss_and_grads(x, model, data_loader):

    # if type(x).__module__ == np.__name__:
    #     x = torch.from_numpy(x).float()
    #     x = x.cuda()

    model.eval()

    x_start = 0
    for p in model.parameters():
        param_size = p.data.size()
        param_idx = 1
        for s in param_size:
            param_idx *= s
        x_part = x[x_start : x_start + param_idx]
        p.data = torch.Tensor(x_part.reshape(param_size))
        x_start += param_idx

    batch_losses = []
    batch_grads = []
    for it, batch in enumerate(data_loader):

        # Move data to correct device
        # inputs = inputs.to(device)
        # targets = targets.to(device)

        with torch.set_grad_enabled(True):
            # loss, grads = model(idx=inputs, targets=targets, grads=True)
            loss, grads = model._step(batch=batch, batch_idx=1, train=True, grads=True)

        # Todo: average over dataset
        batch_losses.append(loss)
        # batch_grads.append(None if grads is None else grads.cpu().numpy().astype(np.float64))
        batch_grads.append(None if grads is None else grads)

    mean_losses = torch.mean(torch.stack(batch_losses))
    mean_grads = torch.mean(torch.stack(batch_grads), dim=0)

    return (mean_losses, mean_grads.cpu().numpy().astype(np.float64))


def get_weights(model):
    """
    Given a model, return a vector of weights.
    """
    x0 = None
    for p in model.parameters():
        if x0 is None:
            x0 = p.data.view(-1)
        else:
            x0 = torch.cat((x0, p.data.view(-1)))
    return x0.cpu().numpy()


def get_sharpness(data_loader, model, subspace_dim=10, epsilon=1e-3, maxiter=10):
    """
    Compute the sharpness around some point in weight space, as specified
    in Keskar et. al. (2016) Sec 2.2.2:
    https://arxiv.org/pdf/1609.04836.pdf

    See:
        https://gist.github.com/arthurmensch/c55ac413868550f89225a0b9212aa4cd
        https://gist.github.com/gngdb/a9f912df362a85b37c730154ef3c294b
        https://github.com/keskarnitish/large-batch-training
        https://github.com/wenwei202/smoothout
        https://github.com/keras-team/keras/pull/3064
    """

    x0 = get_weights(model)

    f_x0, _ = get_loss_and_grads(x0, model, data_loader)
    f_x0 = -f_x0
    logging.info("min loss f_x0 = {loss:.4f}".format(loss=f_x0))

    if 0 == subspace_dim:
        x_min = np.reshape(x0 - epsilon * (np.abs(x0) + 1), (x0.shape[0], 1))
        x_max = np.reshape(x0 + epsilon * (np.abs(x0) + 1), (x0.shape[0], 1))
        bounds = np.concatenate([x_min, x_max], 1)
        func = lambda x: get_loss_and_grads(x, model, data_loader)
        init_guess = x0
    else:
        assert subspace_dim <= x0.shape[0]

        # Computed via Keskar, et. al
        # https://arxiv.org/pdf/1609.04836.pdf

        A_plus = np.random.rand(subspace_dim, x0.shape[0]) * 2.0 - 1.0
        A_plus_norm = np.linalg.norm(A_plus, axis=1)
        A_plus = A_plus / np.reshape(A_plus_norm, (subspace_dim, 1))
        A = np.linalg.pinv(A_plus)

        abs_bound = epsilon * (np.abs(np.dot(A_plus, x0)) + 1)
        abs_bound = np.reshape(abs_bound, (abs_bound.shape[0], 1))
        bounds = np.concatenate([-abs_bound, abs_bound], 1)

        def func(y):
            f_loss, f_grads = get_loss_and_grads(
                x0 + np.dot(A, y),
                model,
                data_loader,
            )
            return f_loss, np.dot(np.transpose(A), f_grads)

        init_guess = np.zeros(subspace_dim)

    minimum_x, f_x, d = scipy.optimize.fmin_l_bfgs_b(
        func,
        init_guess,
        maxiter=maxiter,
        bounds=bounds,
        disp=1,
    )
    f_x = -f_x
    logging.info("max loss f_x = {loss:.4f}".format(loss=f_x))

    # Eq 4 in Keskar
    phi = (f_x - f_x0) / (1 + f_x0) * 100

    # Restore parameter values
    x0 = torch.from_numpy(x0).float()
    # x0 = x0.cuda()
    x_start = 0
    for p in model.parameters():
        param_size = p.data.size()
        param_idx = 1
        for s in param_size:
            param_idx *= s
        x_part = x0[x_start : x_start + param_idx]
        p.data = x_part.view(param_size)
        x_start += param_idx

    return phi
