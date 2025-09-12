import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SEED = 42
rng = np.random.default_rng(SEED)

mu_A = np.zeros(5)
Sigma_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])

mu_B = np.full(5, 1.5)
Sigma_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

nA = 500
nB = 500

XA = rng.multivariate_normal(mu_A, Sigma_A, size=nA)
XB = rng.multivariate_normal(mu_B, Sigma_B, size=nB)
X = np.vstack([XA, XB])
y = np.array([0]*nA + [1]*nB)

pca = PCA(n_components=2, random_state=SEED)
X2 = pca.fit_transform(X)
evr = pca.explained_variance_ratio_

plt.figure(figsize=(7,5))
plt.scatter(X2[y==0,0], X2[y==0,1], s=14, alpha=0.7, label="Class A")
plt.scatter(X2[y==1,0], X2[y==1,1], s=14, alpha=0.7, label="Class B")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"PCA (2D) — A vs B | Var. explained: PC1={evr[0]:.2f}, PC2={evr[1]:.2f}")
plt.legend()
plt.tight_layout()
plt.show()