import numpy as np
import matplotlib.pyplot as plt

SEED = 42
N_CLASSES = 4
N_PER_CLASS = 100

means = np.array([
    [2.0, 3.0],
    [5.0, 6.0],
    [8.0, 1.0],
    [15.0, 4.0],
])
stds = np.array([
    [0.8, 2.5],
    [1.2, 1.9],
    [0.9, 0.9],
    [0.5, 2.0],
])

def generate_data(means: np.ndarray, stds: np.ndarray, n_per_class: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for c in range(len(means)):
        x = rng.normal(loc=means[c, 0], scale=stds[c, 0], size=n_per_class)
        y = rng.normal(loc=means[c, 1], scale=stds[c, 1], size=n_per_class)
        X_list.append(np.column_stack([x, y]))
        y_list.append(np.full(n_per_class, c, dtype=int))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y

def plot_scatter_with_linear_cuts(X: np.ndarray, y: np.ndarray, means: np.ndarray, save_path: str | None = None):
    plt.figure(figsize=(8, 5))
    for c in np.unique(y):
        pts = X[y == c]
        plt.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.7, label=f"Class {c}")

    cuts = [
        (means[0, 0] + means[1, 0]) / 2.0,
        (means[1, 0] + means[2, 0]) / 2.0,
        (means[2, 0] + means[3, 0]) / 2.0,
    ]
    for cx in cuts:
        plt.axvline(cx, linestyle="--", linewidth=1.5)

    plt.title("Synthetic 2D dataset with suggested linear boundaries")
    plt.xlabel("Feature 1 (x)")
    plt.ylabel("Feature 2 (y)")
    plt.legend(loc="best")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def main():
    X, y = generate_data(means, stds, n_per_class=N_PER_CLASS, seed=SEED)

    plot_scatter_with_linear_cuts(X, y, means, save_path=None)

if __name__ == "__main__":
    main()