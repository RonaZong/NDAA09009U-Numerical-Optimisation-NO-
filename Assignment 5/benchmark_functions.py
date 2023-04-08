import numpy as np


def ellipsoid(x, alpha=1000):
    result = 0
    if isinstance(x, (list, np.ndarray)):
        d = len(x)
        for i in range(d):
            result = result + alpha**(i/(d-1))*x[i]**2
        return result
    else:
        result = x**2
        return result


def ellipsoid_gradient(x, alpha=1000):
    if isinstance(x, (list, np.ndarray)):
        d = len(x)
        result = np.zeros(shape=d)
        for i, xi in enumerate(x, start=1):
            result[i-1] = alpha**((i-1)/(d-1))*2*xi
        return result
    else:
        return 2*x


def ellipsoid_hessian(x, alpha=1000):
    if isinstance(x, (list, np.ndarray)):
        d = len(x)
        result = np.zeros(shape=(d, d))
        for i in range(0, d):
            result[i, i] = alpha**(i/(d-1))*2
        return result
    else:
        return 2


def rosenbrock_banana(x):
    x1 = x[0]
    x2 = x[1]
    result = (1-x1) ** 2 + 100*(x2-x1**2)**2
    return result


def rosenbrock_banana_gradient(x):
    x1 = x[0]
    x2 = x[1]
    gradient_1 = -2*(1-x1)-400*x1*(x2-x1**2)
    gradient_2 = 200*(x2-x1**2)
    result = np.array([gradient_1, gradient_2])
    return result


def rosenbrock_banana_hessian(x):
    x1 = x[0]
    x2 = x[1]
    hessian_1_x1 = 2-400*(x2-3*x1**2)
    hessian_1_x2 = -400*x1
    hessian_2_x1 = -400*x1
    hessian_2_x2 = 200
    result = np.array([[hessian_1_x1, hessian_1_x2],
                      [hessian_2_x1, hessian_2_x2]])
    return result


def log_ellipsoid(x, epsilon=10**(-4), alpha=1000):
    result = np.log(epsilon + ellipsoid(x, alpha))
    return result


def log_ellipsoid_gradient(x, epsilon=10**(-4), alpha=1000):
    if isinstance(x, (list, np.ndarray)):
        d = len(x)
        result = np.zeros(shape=d)
        for i, xi in enumerate(x, start=1):
            result[i-1] = (alpha**((i-1)/(d-1))*2*xi) / \
                (ellipsoid(x, alpha)+epsilon)
        return result
    else:
        return (2*x) / (ellipsoid(x, alpha)+epsilon)


def log_ellipsoid_hessian(x, epsilon=10**(-4), alpha=1000):
    temp = ellipsoid(x, alpha)+epsilon
    if isinstance(x, (list, np.ndarray)):
        d = len(x)
        result = np.zeros(shape=(d, d))
        for i in range(0, d):
            for j in range(0, d):
                if i == j:
                    result[i, j] = (2*alpha**(i/(d-1))*temp -
                                    (2*alpha**(i/(d-1))*x[i])**2) / temp**2
                else:
                    result[i, j] = -1*(2*alpha**(i/(d-1))*x[i]) * \
                        (2*alpha**(j/(d-1))*x[j]) / temp**2
        return result
    else:
        return (2*temp - 4*x**2) / (temp**2)


def h(x, q):
    return (np.log(1 + np.exp(-np.abs(q*x))) + np.where(q*x < 0, 0, q*x))/q


def attractive_sector(x, q=10**4):
    result = 0
    if isinstance(x, (list, np.ndarray)):
        d = len(x)
        for i in range(d):
            result = result + h(x[i], q)**2 + 100*h(-x[i], q)**2
        return result
    else:
        return h(x, q)**2 + 100*h(-x, q)**2


def h_gradient(x, q):
    if x >= 0:
        return 1 - np.exp(-q*x)/(1+np.exp(-q*x))
    else:
        return np.exp(q*x)/(1+np.exp(q*x))


def attractive_sector_gradient(x, q=10**4):
    if isinstance(x, (list, np.ndarray)):
        d = len(x)
        result = np.zeros(shape=d)
        for i, xi in enumerate(x):
            result[i] = 2*h_gradient(xi, q)*h(xi, q) - \
                200*h_gradient(-xi, q)*h(-xi, q)
        return result
    else:
        return 2*h_gradient(x, q)*h(x, q)-200*h_gradient(-x, q)*h(-x, q)


def h_hessian(x, q):
    if x >= 0:
        return (q*np.exp(-q*x))/(1+np.exp(-q*x))**2
    else:
        return (q*np.exp(q*x))/(1+np.exp(q*x))**2


def attractive_sector_hessian(x, q=10**4):
    if isinstance(x, (list, np.ndarray)):
        d = len(x)
        result = np.zeros(shape=(d, d))
        for i in range(d):
            result[i, i] = 2*h_gradient(x[i], q)**2 + 2*h(x[i], q)*h_hessian(
                x[i], q) + 200*h_gradient(-x[i], q)**2 + 200*h(-x[i], q)*h_hessian(-x[i], q)
        return result
    else:
        return 2*h_gradient(x, q)**2 + 2*h(x, q)*h_hessian(x, q) + 200*h_gradient(-x, q)**2 + 200*h(-x, q)*h_hessian(-x, q)


def sum_different_powers(x):
    result = 0
    if isinstance(x, (list, np.ndarray)):
        d = len(x)
        for i in range(d):
            result = result + x[i]**(2+2*i/(d-1))
        return result
    else:
        return x**2


def sum_different_powers_gradient(x):
    if isinstance(x, (list, np.ndarray)):
        d = len(x)
        result = np.zeros(shape=d)
        for i, xi in enumerate(x):
            result[i] = 2*(1+i/(d-1))*xi**(1+2*i/(d-1))
        return result
    else:
        return 2*x


def sum_different_powers_hessian(x):
    if isinstance(x, (list, np.ndarray)):
        d = len(x)
        result = np.zeros(shape=(d, d))
        for i in range(0, d):
            result[i, i] = 2*(d+i-1)*(d+2*i-1)/(d-1)**2*x[i]**(2*i/(d-1))
        return result
    else:
        return 0
