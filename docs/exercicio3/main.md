## Exercício 1

### Setup

```
import numpy as np

x = np.array([0.5, -0.2])
y = 1.0

W1 = np.array([[0.3, -0.1],
               [0.2,  0.4]])
b1 = np.array([0.1, -0.2])

W2 = np.array([0.5, -0.3])
b2 = 0.2

eta = 0.1
```

### Forward Pass
```
z1 = W1 @ x + b1
h1 = tanh(z1)
u2 = float(W2 @ h1 + b2)
y_hat = float(tanh(u2))
L = (y - y_hat)**2
```

### Backward Pass
```
dL_dyhat = 2 * (y_hat - y)
dL_du2   = dL_dyhat * (1 - np.tanh(u2)**2)

dL_dW2 = dL_du2 * h1
dL_db2 = dL_du2

dL_dh1 = dL_du2 * W2
dL_dz1 = dL_dh1 * (1 - np.tanh(z1)**2)

dL_dW1 = np.outer(dL_dz1, x)
dL_db1 = dL_dz1
```

Gradientes:
=== Backward Pass (Gradients) ===

dL/dy_hat     = -1.26550687

dL/du2        = -1.09482791

dL/dW2        = [-0.28862383  0.19496791]

dL/db2        = -1.09482791

dL/dh1        = [-0.54741396  0.32844837]

dL/dz1        = [-0.50936975  0.31803236]

dL/dW1        =
[[-0.25468488  0.10187395]
 [ 0.15901618 -0.06360647]]

dL/db1        = [-0.50936975  0.31803236]

### Atualização dos Parâmetros
W2_new = W2 - eta * dL_dW2
b2_new = b2 - eta * dL_db2
W1_new = W1 - eta * dL_dW1
b1_new = b1 - eta * dL_db1

=== Parameter Update (eta = 0.1) ===

W2_new = [ 0.52886238 -0.31949679]

b2_new = 0.30948279

W1_new =
[[ 0.32546849 -0.1101874 ]
[ 0.18409838  0.40636065]]

b1_new = [ 0.15093698 -0.23180324]

### Análise e Resultados
- O forward pass mostrou que a rede previu ŷ ≈ 0.367, abaixo do target real (1.0).
- O erro quadrático foi ≈ 0.40, indicando diferença relevante.
- O backward pass forneceu os gradientes para cada peso e viés.
- A atualização dos parâmetros deslocou os pesos e vieses no sentido de reduzir o erro.

## Exercício 2

### Setup
```
import numpy as np
import matplotlib.pyplot as plt

mean0 = np.array([2.5, 2.5])
mean1 = np.array([4.0, 4.0])

cov0 = np.array([[1.5, 0.0],
                 [0.0, 1.5]])
cov1 = np.array([[1.5, 0.0],
                 [0.0, 1.5]])

n_per_class = 1000
rng = np.random.default_rng(123)

X0 = rng.multivariate_normal(mean0, cov0, size=n_per_class)
X1 = rng.multivariate_normal(mean1, cov1, size=n_per_class)

X = np.vstack([X0, X1])
y = np.hstack([-np.ones(n_per_class), np.ones(n_per_class)])
```

### Implementação
```
def perceptron_train(X, y, eta=0.01, max_epochs=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    w = np.zeros(X.shape[1])
    b = 0.0

    for epoch in range(max_epochs):
        updates = 0
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        for xi, yi in zip(X, y):
            if yi * (w @ xi + b) <= 0:
                w += eta * yi * xi
                b += eta * yi
                updates += 1
        if updates == 0:
            break
    return w, b
```

### Resultados (melhor execução entre 5 runs)
- Melhor acurácia final: **0.7970**  
- Acurácia média: **0.7689**  
- Pesos finais (w): [0.08021828, 0.08821639]  
- Viés (b): -0.50  
- Épocas: 100 (sem convergência total) 

### Análise e Resultados
- O dataset apresenta **forte sobreposição** → o perceptron não converge totalmente.  
- A acurácia estabilizou em ~80%.  
- O modelo linear não consegue separar os pontos sobrepostos.  
- Conclusão: é necessário **modelos não lineares (ex. MLP)** para atingir separação melhor.  

---

## Exercício 3

### Setup
```
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=1500, n_features=4, n_informative=4,
    n_redundant=0, n_classes=3, n_clusters_per_class=2,
    class_sep=1.2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)
```

### Modelo
```
MLP([4, 16, 3])
Ativação: ReLU nas ocultas
Saída: Softmax
Loss: Cross-Entropy
Otimizador: SGD (lr=0.05, batch=64, 200 épocas)
```

### Resultados
```
Acurácia Treino ≈ 0.93
Acurácia Teste  ≈ 0.89
```

### Visualizações
- Curva de Loss por Época  
- Matriz de Confusão (3 classes)  

### Análise e Resultados
- O modelo aprendeu bem as fronteiras de decisão.  
- A padronização das features estabilizou o treino.  
- A rede rasa (1 camada oculta) já generaliza bem, mas há margem para melhora.  
- Pequena diferença entre treino e teste indica bom equilíbrio entre viés e variância.  

---

## Exercício 4

### Setup
```
Mesmo dataset do Exercício 3

Arquitetura: [4, 32, 16, 3]
Ativações: ReLU (ocultas), Softmax (saída)
Loss: Cross-Entropy
Otimizador: SGD (lr=0.03, wd=1e-4)
Épocas: 250
Batch: 64
```

### Resultados
```
Acurácia Treino ≈ 0.95
Acurácia Teste  ≈ 0.90
```

### Visualizações
- Loss de treino por época (decrescendo de forma suave)
- Matriz de confusão mais “limpa” que no Ex.3

### Análise e Resultados
- A rede mais profunda obteve leve melhora de desempenho.  
- A segunda camada oculta aumenta a **capacidade de representação**.  
- A regularização (`weight decay`) reduziu overfitting.  
- A curva de loss mostrou convergência estável.  
- Conclusão: a maior profundidade permitiu **aprender fronteiras mais complexas**, com boa generalização.  

---

## Conclusão Geral
- **Ex1:** Implementação manual de MLP — entendimento da mecânica do gradiente.  
- **Ex2:** Perceptron em dados sobrepostos — limitações de linearidade.  
- **Ex3:** MLP simples — primeira generalização não linear bem-sucedida.  
- **Ex4:** MLP mais profundo — melhora de capacidade e estabilidade.  