import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np


def gradient_penalty(D, kernel, fake_data, real_data):
    bs = min(len(real_data), len(fake_data))
    real_data, fake_data = real_data[:bs], fake_data[:bs]
    
    # Create random weights for interpolation
    alpha = torch.rand(bs, 1, 1, 1, device=real_data.device)
    
    # Interpolate between real and fake data
    interp = (1. - alpha) * real_data + alpha * fake_data
    
    # Compute the discriminator output for interpolated data
    x_hat = D(interp)
    
    # Function to compute kernel mean for given data
    def Ekx(yy):
        return torch.mean(kernel(x_hat, yy, K_XY_only=True), dim=1)
    
    Ekxr, Ekxf = Ekx(D(real_data)), Ekx(D(fake_data))
    witness = Ekxr - Ekxf
    
    # Compute gradients of the witness function w.r.t. interpolated data
    gradients = autograd.grad(
        outputs=witness, inputs=interp,
        grad_outputs=torch.ones_like(witness),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    
    return torch.mean((gradients_norm - 1) ** 2)

def _mix_rq_kernel(X, Y, alphas=[.2, .5, 1., 2., 5.], wts=None, K_XY_only=False, add_dot=.1):

    if wts is None:
        wts = torch.tensor([1.] * len(alphas))

    XX = torch.mm(X, X.t())
    XY = torch.mm(X, Y.t())
    YY = torch.mm(Y, Y.t())

    X_sqnorms = torch.diag(XX)
    Y_sqnorms = torch.diag(YY)

    r = lambda x: x.unsqueeze(0)
    c = lambda x: x.unsqueeze(1)

    K_XX, K_XY, K_YY = 0., 0., 0.

    XYsqnorm = torch.maximum(-2. * XY + c(X_sqnorms) + r(Y_sqnorms), torch.zeros_like(XY))

    for alpha, wt in zip(alphas, wts):
        logXY = torch.log(1. + XYsqnorm / (2. * alpha))
        K_XY += wt * torch.exp(-alpha * logXY)
    if add_dot > 0:
        K_XY += add_dot * XY

    if K_XY_only:
        return K_XY

    XXsqnorm = torch.maximum(-2. * XX + c(X_sqnorms) + r(X_sqnorms), torch.zeros_like(XX))
    YYsqnorm = torch.maximum(-2. * YY + c(Y_sqnorms) + r(Y_sqnorms), torch.zeros_like(YY))

    for alpha, wt in zip(alphas, wts):
        logXX = torch.log(1. + XXsqnorm / (2. * alpha))
        logYY = torch.log(1. + YYsqnorm / (2. * alpha))
        K_XX += wt * torch.exp(-alpha * logXX)
        K_YY += wt * torch.exp(-alpha * logYY)
    if add_dot > 0:
        K_XX += add_dot * XX
        K_YY += add_dot * YY

    wts = torch.sum(wts.float())
    return K_XX, K_XY, K_YY, wts


def _mix_rbf_kernel(X, Y, sigmas=[2.0, 5.0, 10.0, 20.0, 40.0, 80.0], wts=None, 
                    K_XY_only=False):
    if wts is None:
        wts = [1.] * len(sigmas)

    XX = torch.mm(X, X.t())
    XY = torch.mm(X, Y.t())
    YY = torch.mm(Y, Y.t())
        
    X_sqnorms = torch.diag(XX)
    Y_sqnorms = torch.diag(YY)

    r = lambda x: x.unsqueeze(0)
    c = lambda x: x.unsqueeze(1)

    K_XX, K_XY, K_YY = 0., 0., 0.
    
    XYsqnorm = -2 * XY + c(X_sqnorms) + r(Y_sqnorms)
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XY += wt * torch.exp(-gamma * XYsqnorm)
        
    if K_XY_only:
        return K_XY
    
    XXsqnorm = -2 * XX + c(X_sqnorms) + r(X_sqnorms)
    YYsqnorm = -2 * YY + c(Y_sqnorms) + r(Y_sqnorms)
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * torch.exp(-gamma * XXsqnorm)
        K_YY += wt * torch.exp(-gamma * YYsqnorm)
        
    return K_XX, K_XY, K_YY, torch.sum(torch.tensor(wts))

def mysqrt(x):
    """Safely compute the square root to avoid NaN gradients for negative inputs."""
    return torch.sqrt(torch.relu(x))

def _distance_kernel(X, Y, K_XY_only=False):
    XX = torch.mm(X, X.t())
    XY = torch.mm(X, Y.t())
    YY = torch.mm(Y, Y.t())
        
    X_sqnorms = torch.diag(XX)
    Y_sqnorms = torch.diag(YY)

    r = lambda x: x.unsqueeze(0)
    c = lambda x: x.unsqueeze(1)

    # Compute the distance-based kernel matrices
    K_XY = c(mysqrt(X_sqnorms)) + r(mysqrt(Y_sqnorms)) - mysqrt(-2 * XY + c(X_sqnorms) + r(Y_sqnorms))

    if K_XY_only:
        return K_XY

    K_XX = c(mysqrt(X_sqnorms)) + r(mysqrt(X_sqnorms)) - mysqrt(-2 * XX + c(X_sqnorms) + r(X_sqnorms))
    K_YY = c(mysqrt(Y_sqnorms)) + r(mysqrt(Y_sqnorms)) - mysqrt(-2 * YY + c(Y_sqnorms) + r(Y_sqnorms))
        
    return K_XX, K_XY, K_YY, False

def _mmd2(K_XX, K_XY, K_YY):
    m = K_XX.shape[0]  # Number of samples in X
    n = K_YY.shape[0]  # Number of samples in Y
    trace_X = torch.trace(K_XX)
    trace_Y = torch.trace(K_YY)
    mmd2 = ((torch.sum(K_XX) - trace_X) / (m * (m - 1)) +
            (torch.sum(K_YY) - trace_Y) / (n * (n - 1)) -
            2 * torch.sum(K_XY) / (m * n))
    return mmd2

def mmd2(kernel, x, y):
    K_XX, K_XY, K_YY, _ = kernel(x, y)
    return _mmd2(K_XX, K_XY, K_YY)