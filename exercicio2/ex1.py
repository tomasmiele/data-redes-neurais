import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

mean0 = np.array([1.5, 1.5])
cov0  = np.array([[0.5, 0.0],
                  [0.0, 0.5]])

mean1 = np.array([5.0, 5.0])
cov1  = np.array([[0.5, 0.0],
                  [0.0, 0.5]])

n = 1000
X0 = rng.multivariate_normal(mean0, cov0, size=n)
X1 = rng.multivariate_normal(mean1, cov1, size=n)

X = np.vstack([X0, X1])
y = np.hstack([-np.ones(n), np.ones(n)])

perm = rng.permutation(len(X))
X, y = X[perm], y[perm]

w = np.zeros(2)
b = 0.0
eta = 0.01
max_epochs = 100

def predict_scores(X, w, b):
    return X @ w + b

def predict_labels(X, w, b):
    return np.where(predict_scores(X, w, b) >= 0.0, 1.0, -1.0)

acc_history = []
for epoch in range(max_epochs):
    updates = 0
    for xi, yi in zip(X, y):
        if yi * (w @ xi + b) <= 0.0:
            w += eta * yi * xi
            b += eta * yi
            updates += 1

    y_hat = predict_labels(X, w, b)
    acc = (y_hat == y).mean()
    acc_history.append(acc)

    if updates == 0:
        break

print(f"Pesos finais: {w}")
print(f"Viés final: {b}")
print(f"Épocas: {len(acc_history)}  |  Acurácia final: {acc_history[-1]:.4f}")

plt.figure(figsize=(6, 5))
plt.scatter(X[y==-1][:,0], X[y==-1][:,1], s=12, label="Classe 0 (-1)")
plt.scatter(X[y== 1][:,0], X[y== 1][:,1], s=12, label="Classe 1 (+1)")

x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
if abs(w[1]) > 1e-12:
    xs = np.linspace(x_min, x_max, 300)
    ys = -(w[0]/w[1]) * xs - b / w[1]
    plt.plot(xs, ys, linewidth=2, label="Fronteira de decisão")
else:
    x_line = -b / (w[0] + 1e-12)
    plt.axvline(x_line, linewidth=2, label="Fronteira de decisão")

mis_idx = np.where(predict_labels(X, w, b) != y)[0]
if len(mis_idx) > 0:
    plt.scatter(X[mis_idx,0], X[mis_idx,1], marker='x', s=35, label="Misclassificados")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Dados e Fronteira de Decisão (Perceptron)")
plt.legend()
plt.tight_layout()
plt.show()

# (b) Acurácia por época
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(acc_history)+1), acc_history, marker='o')
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.title("Acurácia de Treino por Época")
plt.ylim(0, 1.05)
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()