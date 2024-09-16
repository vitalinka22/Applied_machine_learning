
import random
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


# loss function
def loss_scalar(theta_0: np.ndarray, theta_1: np.ndarray) -> float:
    # evaluate loss function for set of weights theta
    return theta_0 ** 2 + 2 * theta_1 ** 2


def loss(theta: np.ndarray) -> float:
    # evaluate loss function for set of weights theta
    if theta.ndim == 1:
        _loss = theta[0] ** 2 + 2 * theta[1] ** 2
    elif theta.ndim == 2:
        _loss = theta[:, 0] ** 2 + 2 * theta[:, 1] ** 2
    return _loss


def nabla_loss(theta: np.ndarray) -> np.ndarray:
    # return gradient at theta
    diff_theta_0 = 2 * theta[0]
    diff_theta_1 = 4 * theta[1]

    return np.array([diff_theta_0, diff_theta_1])


# generate some sampling points that we can use for visualization
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
print(f'shape of X: {X.shape}')
print(f'shape of Y: {Y.shape}')

# display loss surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
zs = np.array(loss_scalar(X.flatten(), Y.flatten()))
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z)
ax.set_xlabel(r'parameter $\theta_0$')
ax.set_ylabel(r'parameter $\theta_1$')
ax.set_zlabel(r'loss $\mathcal{L}$')
plt.show()

"""
Optimization using GD-based methods
"""

theta_0 = np.array([-1, 2])  # starting point

def gradient_descent_with_momentum(f, df, x0: np.ndarray, lr: float = 0.1, beta: float = 0.5):
    reltol = 10 ** (-6)  # improvement of function value

    # initialization
    converged, step = False, 0
    v=np.zeros_like(x0)
    x, x_steps, loss_steps = x0, [x0], []

    while (not converged) and (step < 500):
        print(f'iteration step {step + 1}')
        y = f(x)  # current function value
        loss_steps.append(y)

        v = beta*v + (1 - beta)*df(x)
        x = x - lr * v  # update

        # relative convergence criterion
        y_next = f(x)  # new function value
        if np.abs(y - y_next) < reltol:
            converged = True
            print(f'converged at {x}')

        if step == 499:
            print(f'stopped at max. number of iterations!')

        step += 1
        x_steps.append(x)


    x_min = x_steps[-1]
    return x_min, np.array(x_steps), np.array(loss_steps)
def gradient_descent(f, df, x0: np.ndarray, lr: float = 0.1):
    reltol = 10 ** (-6)  # improvement of function value

    # initialization
    converged, step = False, 0
    x, x_steps, loss_steps = x0, [x0], []

    while (not converged) and (step < 500):
        print(f'iteration step {step + 1}')
        y = f(x)  # current function value
        loss_steps.append(y)

        x = x - lr * df(x)  # update

        # relative convergence criterion
        y_next = f(x)  # new function value
        if np.abs(y - y_next) < reltol:
            converged = True
            print(f'converged at {x}')

        if step == 499:
            print(f'stopped at max. number of iterations!')

        step += 1
        x_steps.append(x)


    x_min = x_steps[-1]
    return x_min, np.array(x_steps), np.array(loss_steps)

def gradient_descent_adagrad (f, df, x0: np.ndarray, lr: float = 0.5):
    reltol = 10 ** (-6)  # improvement of function value

    # initialization
    converged, step = False, 0
    x, x_steps, loss_steps = x0, [x0], []
    s = np.zeros_like(x0)

    while (not converged) and (step < 500):
        print(f'iteration step {step + 1}')
        y = f(x)  # current function value
        loss_steps.append(y)
        s = s +(df(x) ** 2)
        x = x - lr * df(x)/(np.sqrt(s) + reltol)   # update

        # relative convergence criterion
        y_next = f(x)  # new function value
        if np.abs(y - y_next) < reltol:
            converged = True
            print(f'converged at {x}')

        if step == 499:
            print(f'stopped at max. number of iterations!')

        step += 1
        x_steps.append(x)


    x_min = x_steps[-1]
    return x_min, np.array(x_steps), np.array(loss_steps)

def gradient_descent_adam (f, df, x0: np.ndarray, lr: float = 0.95, beta_1 = 0.9, beta_2 = 0.999):
    reltol = 10 ** (-6)  # improvement of function value

    # initialization
    converged, step = False, 0
    x, x_steps, loss_steps = x0, [x0], []
    s = np.zeros_like(x0)
    v = np.zeros_like(x0)

    while (not converged) and (step < 500):
        print(f'iteration step {step + 1}')
        y = f(x)  # current function value
        loss_steps.append(y)
        v = beta_1 * v + (1 - beta_1) * df(x)
        s = beta_2 * s + (1 - beta_2)*df(x)**2
        v = v/(1 - beta_1)
        s = s/(1 - beta_2)
        x = x - lr * v/(np.sqrt(s) + reltol)   # update

        # relative convergence criterion
        y_next = f(x)  # new function value
        if np.abs(y - y_next) < reltol:
            converged = True
            print(f'converged at {x}')

        if step == 499:
            print(f'stopped at max. number of iterations!')

        step += 1
        x_steps.append(x)


    x_min = x_steps[-1]
    return x_min, np.array(x_steps), np.array(loss_steps)


# compare algorithms
theta_min_gd, theta_iters_gd, loss_iters_gd = gradient_descent(f=loss, df=nabla_loss, x0=theta_0)
theta_min_gdm, theta_iters_gdm, loss_iters_gdm = gradient_descent_with_momentum(f=loss, df=nabla_loss, x0=theta_0)
theta_min_adagrad, theta_iters_adagrad, loss_iters_adagrad = gradient_descent_adagrad(f=loss, df=nabla_loss, x0=theta_0)
theta_min_adam, theta_iters_adam, loss_iters_adam = gradient_descent_adam(f=loss, df=nabla_loss, x0=theta_0)

plt.figure()
plt.plot(loss_iters_gd, color='red', marker='o', linestyle='-', label='GD')
plt.plot(loss_iters_gdm, color='blue', marker='x', linestyle='-', label='GDM')
plt.plot(loss_iters_adagrad, color='yellow', marker='x', linestyle='-', label='AdaGrad')
plt.plot(loss_iters_adam, color='pink', marker='x', linestyle='-', label='Adam')
plt.xlabel('iteration')
plt.ylabel(r'loss $\mathcal{L}$')
# plt.yscale('log')
plt.show()

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(theta_iters_gd[:, 0], theta_iters_gd[:, 1], loss(theta_iters_gd), color='red', marker='.', label='GD')
ax.plot(theta_iters_gdm[:, 0], theta_iters_gdm[:, 1], loss(theta_iters_gdm), color='blue', marker='.', label='GDM')
ax.plot(theta_iters_adagrad[:, 0], theta_iters_adagrad[:, 1], loss(theta_iters_adagrad), color='yellow', marker='.', label='AdaGrad')
ax.plot(theta_iters_adam[:, 0], theta_iters_adam[:, 1], loss(theta_iters_adam), color='pink', marker='.', label='Adam')
ax.plot_surface(X, Y, Z, cmap='Greys', edgecolor='none', alpha=0.5)
ax.set_xlabel(r'parameter $\theta_0$')
ax.set_ylabel(r'parameter $\theta_1$')
ax.set_zlabel(r'loss $\mathcal{L}$')
plt.show()

plt.figure()
plt.contourf(X, Y, Z, levels=50, cmap='Greys')
plt.plot(theta_iters_gd[:, 0], theta_iters_gd[:, 1], color='red', marker='.', label='GD')
plt.plot(theta_iters_gdm[:, 0], theta_iters_gdm[:, 1], color='blue', marker='.', label='GDM')
plt.plot(theta_iters_adagrad[:, 0], theta_iters_adagrad[:, 1], color='yellow', marker='.', label='AdaGrad')
plt.plot(theta_iters_adam[:, 0], theta_iters_adam[:, 1], color='pink', marker='.', label='Adam')
cb = plt.colorbar()
cb.set_label(r'loss $\mathcal{L}$')
plt.xlabel(r'parameter $\theta_0$')
plt.ylabel(r'parameter $\theta_1$')
plt.legend()
plt.show()