## Objetivo

Essa atividade foi elaborada para testar suas habilidades em gerar conjuntos de dados sintéticos, lidar com desafios de dados do mundo real e preparar dados para serem alimentados em redes neurais.

## Exercício 1

### Generate the Data

``` 
import numpy as np

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

X, y = generate_data(means, stds, n_per_class=N_PER_CLASS, seed=SEED)

print(X, y)
```

### Plot the Data

```
import matplotlib.pyplot as plt

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

plot_scatter_with_linear_cuts(X, y, means, save_path=None)
```
![Output do código](./data.png)

### Analyse and Draw Boundaries

**(a) Distribuição e sobreposição**

- **Classe 0 (μ ≈ [2,3])**: mais alongada no eixo *y* (σy = 2.5), formando uma “coluna” vertical à esquerda.  
- **Classe 1 (μ ≈ [5,6])**: variância moderada, localizada acima da Classe 2 em *y*, separada da Classe 0 em *x*.  
- **Classe 2 (μ ≈ [8,1])**: mais compacta e abaixo da Classe 1 em *y*.  
- **Classe 3 (μ ≈ [15,4])**: bem à direita, com σy = 2.0, praticamente sem mistura com as demais.  

**(b) Uma fronteira linear simples separa todas as classes?**

Sim. Um **conjunto de fronteiras lineares (retas verticais)** é suficiente para separar bem as quatro classes, pois os clusters estão ordenados principalmente ao longo do eixo *x*.  

![Data com barreiras](./data-with-boundaries.png)

**(c) Fronteiras sugeridas**

As fronteiras de decisão podem ser esboçadas como linhas verticais em:  

- **x ≈ 3.5** (entre Classe 0 e Classe 1)  
- **x ≈ 6.5** (entre Classe 1 e Classe 2)  
- **x ≈ 11.5** (entre Classe 2 e Classe 3)

## Exercício 2

### Generate the Data

```
import numpy as np

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

print(X.shape, y.shape)
```

### Apply PCA and Plot the Data

```
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
```

![Output do PCA](./pca.png)

### Analyse the Plots

**(a) Relação entre as classes na projeção 2D**

- As classes aparecem **deslocadas ao longo de PC1** (efeito do vetor de médias).  
- Há **sobreposição visível** no centro: regiões em que pontos das duas classes se misturam.  
- As covariâncias diferentes dão orientações e alongamentos distintos aos clusters.  

**(b) Linear separability**

- Um corte linear em PC1 poderia reduzir parte do erro, mas **não separa perfeitamente** A e B.  
- A estrutura é **não linear**, devido à sobreposição e curvatura nas combinações de variáveis originais (5D).  
- **Modelos lineares simples** (perceptron, regressão logística) terão dificuldade.  

**(c) Por que usar redes neurais**

- Redes neurais com ativações não lineares (ReLU, tanh, sigmoid) podem aprender **fronteiras curvas** adaptando-se às orientações internas dos clusters.  
- Assim, conseguem classificar melhor quando os dados não são linearmente separáveis.