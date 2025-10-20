import numpy as np

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    t = np.tanh(x)
    return 1.0 - t**2

def exercise1(verbose=True):
    x = np.array([0.5, -0.2], dtype=float)
    y = 1.0

    W1 = np.array([[0.3, -0.1],
                   [0.2,  0.4]], dtype=float)
    b1 = np.array([0.1, -0.2], dtype=float)

    W2 = np.array([0.5, -0.3], dtype=float)
    b2 = 0.2

    eta_update = 0.1

    z1 = W1 @ x + b1
    h1 = tanh(z1)
    u2 = float(W2 @ h1 + b2)
    y_hat = float(tanh(u2))
    L = (y - y_hat)**2

    if verbose:
        print("=== Forward Pass ===")
        print(f"x           = {x}")
        print(f"W1          =\n{W1}")
        print(f"b1          = {b1}")
        print(f"z1 = W1@x+b1= {z1}")
        print(f"h1 = tanh(z1)= {h1}")
        print(f"W2          = {W2}")
        print(f"b2          = {b2:.6f}")
        print(f"u2 = W2·h1+b2= {u2:.6f}")
        print(f"y_hat=tanh(u2)= {y_hat:.6f}")
        print("\n=== Loss (MSE) ===")
        print(f"L = (y - y_hat)^2 = {L:.8f}")

    dL_dyhat = 2.0 * (y_hat - y)
    dL_du2 = dL_dyhat * (1.0 - np.tanh(u2)**2)
    dL_dW2 = dL_du2 * h1
    dL_db2 = dL_du2
    dL_dh1 = dL_du2 * W2
    dL_dz1 = dL_dh1 * (1.0 - np.tanh(z1)**2)
    dL_dW1 = np.outer(dL_dz1, x)
    dL_db1 = dL_dz1

    if verbose:
        print("\n=== Backward Pass (Gradients) ===")
        print(f"dL/dy_hat     = {dL_dyhat:.8f}")
        print(f"dL/du2        = {dL_du2:.8f}")
        print(f"dL/dW2        = {dL_dW2}")
        print(f"dL/db2        = {dL_db2:.8f}")
        print(f"dL/dh1        = {dL_dh1}")
        print(f"dL/dz1        = {dL_dz1}")
        print(f"dL/dW1        =\n{dL_dW1}")
        print(f"dL/db1        = {dL_db1}")

    W2_new = W2 - eta_update * dL_dW2
    b2_new = b2 - eta_update * dL_db2
    W1_new = W1 - eta_update * dL_dW1
    b1_new = b1 - eta_update * dL_db1

    if verbose:
        print("\n=== Parameter Update (eta = 0.1) ===")
        print(f"W2_new = {W2_new}")
        print(f"b2_new = {b2_new:.8f}")
        print(f"W1_new =\n{W1_new}")
        print(f"b1_new = {b1_new}")

    return {
        "forward": {"z1": z1, "h1": h1, "u2": u2, "y_hat": y_hat, "L": L},
        "grads": {
            "dL_dyhat": dL_dyhat, "dL_du2": dL_du2,
            "dL_dW2": dL_dW2, "dL_db2": dL_db2,
            "dL_dh1": dL_dh1, "dL_dz1": dL_dz1,
            "dL_dW1": dL_dW1, "dL_db1": dL_db1
        },
        "updated_params": {
            "W1": W1_new, "b1": b1_new, "W2": W2_new, "b2": b2_new
        }
    }

exercise1(verbose=True)