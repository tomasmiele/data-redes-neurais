import numpy as np
import matplotlib.pyplot as plt

mean0 = np.array([2.5, 2.5])
mean1 = np.array([4.0, 4.0])

cov0 = np.array([[1.5, 0.0],
                 [0.0, 1.5]])
cov1 = np.array([[1.5, 0.0],
                 [0.0, 1.5]])

n_per_class = 1000

rng_master = np.random.default_rng(123)

X0 = rng_master.multivariate_normal(mean0, cov0, size=n_per_class)
X1 = rng_master.multivariate_normal(mean1, cov1, size=n_per_class)

X_all = np.vstack([X0, X1])
y_all = np.hstack([-np.ones(n_per_class),
                   +np.ones(n_per_class)])

def perceptron_train(X, y, eta=0.01, max_epochs=100, shuffle=True, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    w = np.zeros(X.shape[1])
    b = 0.0

    def predict_scores(X_): return X_ @ w + b
    def predict_labels(X_): return np.where(predict_scores(X_) >= 0.0, 1.0, -1.0)

    acc_history = []
    updates_per_epoch = []

    for epoch in range(max_epochs):
        updates = 0

        if shuffle:
            idx = rng.permutation(len(X))
            X, y = X[idx], y[idx]

        for xi, yi in zip(X, y):
            if yi * (w @ xi + b) <= 0.0:
                w += eta * yi * xi
                b += eta * yi
                updates += 1

        y_hat = predict_labels(X)
        acc = (y_hat == y).mean()
        acc_history.append(acc)
        updates_per_epoch.append(updates)

        if updates == 0:
            break

    return w, b, np.array(acc_history), len(acc_history), np.array(updates_per_epoch)

runs = 5
results = []
for r in range(runs):
    rng = np.random.default_rng(1000 + r)
    perm = rng.permutation(len(X_all))
    X = X_all[perm].copy()
    y = y_all[perm].copy()

    w, b, acc_hist, epochs, upd_hist = perceptron_train(
        X, y, eta=0.01, max_epochs=100, shuffle=True, rng=rng
    )
    results.append({
        "w": w, "b": b, "acc_hist": acc_hist,
        "epochs": epochs, "updates": upd_hist,
        "final_acc": acc_hist[-1], "rng": 1000 + r,
        "X": X, "y": y
    })

best = sorted(results, key=lambda d: (-d["final_acc"], d["epochs"]))[0]

w, b = best["w"], best["b"]
acc_history = best["acc_hist"]
epochs = best["epochs"]
X, y = best["X"], best["y"]

avg_final_acc = np.mean([r["final_acc"] for r in results])

print("=== Melhores resultados entre runs ===")
print(f"Melhor final_acc: {best['final_acc']:.4f} (rng={best['rng']})")
print(f"Acurácia média das {runs} execuções: {avg_final_acc:.4f}")
print(f"Pesos finais (w): {w}")
print(f"Viés (b): {b}")
print(f"Épocas até parada: {epochs}")

def predict_labels_final(X):
    return np.where(X @ w + b >= 0.0, 1.0, -1.0)

y_pred = predict_labels_final(X)
mis_idx = np.where(y_pred != y)[0]

plt.figure(figsize=(6, 5))
plt.scatter(X[y==-1][:,0], X[y==-1][:,1], s=12, label="Classe 0 (-1)")
plt.scatter(X[y== 1][:,0], X[y== 1][:,1], s=12, label="Classe 1 (+1)")

x_min, x_max = X[:,0].min()-0.8, X[:,0].max()+0.8
if abs(w[1]) > 1e-12:
    xs = np.linspace(x_min, x_max, 400)
    ys = -(w[0]/w[1])*xs - b/w[1]
    plt.plot(xs, ys, linewidth=2, label="Fronteira de decisão")
else:
    x_line = -b/(w[0] + 1e-12)
    plt.axvline(x_line, linewidth=2, label="Fronteira de decisão")

if len(mis_idx) > 0:
    plt.scatter(X[mis_idx,0], X[mis_idx,1], marker='x', s=35, label="Misclassificados")

plt.title("Exercise 2: Dados com Sobreposição e Perceptron")
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(acc_history)+1), acc_history, marker="o")
plt.title("Acurácia de Treino por Época (melhor execução)")
plt.xlabel("Época"); plt.ylabel("Acurácia")
plt.ylim(0, 1.05); plt.grid(True, linestyle=":")
plt.tight_layout()
plt.show()