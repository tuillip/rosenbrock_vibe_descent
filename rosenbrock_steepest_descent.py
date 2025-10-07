
import numpy as np

def rosenbrock(x):
    """Classic 2D Rosenbrock function."""
    x = np.asarray(x, dtype=float)
    return (1 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    """Gradient of the 2D Rosenbrock function."""
    x = np.asarray(x, dtype=float)
    dfdx = -2.0*(1 - x[0]) - 400.0*x[0]*(x[1] - x[0]**2)
    dfdy = 200.0*(x[1] - x[0]**2)
    return np.array([dfdx, dfdy])

def backtracking_line_search(f, grad_f, x, p, alpha0=1.0, c=1e-4, beta=0.5, max_backtracks=30):
    """Armijo backtracking line search."""
    alpha = alpha0
    fx = f(x)
    g = grad_f(x)
    phi0 = fx
    dphi0 = np.dot(g, p)
    for _ in range(max_backtracks):
        if f(x + alpha * p) <= phi0 + c * alpha * dphi0:
            return alpha
        alpha *= beta
    return alpha

def steepest_descent(f, grad_f, x0, tol=1e-8, max_iters=5000, line_search='armijo', alpha0=1.0, verbose=False):
    """Steepest gradient descent with optional Armijo backtracking line search."""
    x = np.asarray(x0, dtype=float)
    path = [x.copy()]
    values = [f(x)]
    for k in range(1, max_iters + 1):
        g = grad_f(x)
        gnorm = np.linalg.norm(g)
        if verbose:
            print(f"iter {k:4d} | f(x)={values[-1]:.6e} | ||grad||={gnorm:.6e} | x={x}")
        if gnorm < tol:
            break
        p = -g  # steepest descent direction
        if line_search == 'armijo':
            alpha = backtracking_line_search(f, grad_f, x, p, alpha0=alpha0)
        else:
            # fixed step (not recommended for Rosenbrock, but available)
            alpha = float(line_search)
        x = x + alpha * p
        path.append(x.copy())
        values.append(f(x))
    return np.array(path), np.array(values), k

def optimize_rosenbrock(x0=(-1.2, 1.0), tol=1e-8, max_iters=5000, alpha0=1.0, verbose=True):
    x0 = np.asarray(x0, dtype=float)
    path, vals, iters = steepest_descent(rosenbrock, grad_rosenbrock, x0, tol=tol, max_iters=max_iters, alpha0=alpha0, verbose=verbose)
    print(f"\nDone in {iters} iterations.")
    print(f"x* ≈ {path[-1]}")
    print(f"f(x*) ≈ {vals[-1]:.10f}")
    return path, vals

if __name__ == '__main__':
    # Demo run
    optimize_rosenbrock(x0=(-1.2, 1.0), tol=1e-8, alpha0=1.0, verbose=True)
