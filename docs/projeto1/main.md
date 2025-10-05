# Projeto 1 - Classification

## 1. Dataset Selection

**Nome do dataset:** Default of Credit Card Clients (Taiwan)  
**Fonte:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients  
**Tamanho:** 30.000 registros, 23 variáveis preditoras + 1 variável-alvo  

**Tarefa:** Classificação binária — prever se um cliente entrará em *default* (inadimplência) no mês seguinte.

**Justificativa da escolha:**
- O problema é **realista e relevante** no contexto financeiro (risco de crédito).  
- Contém **dados mistos** (numéricos e categóricos), o que torna o pré-processamento e o aprendizado mais desafiadores e instrutivos.  
- Possui **volume adequado** (>1.000 amostras e >5 atributos), atendendo aos requisitos do projeto.  
- Apresenta **classes desbalanceadas**, o que permite discutir métricas alternativas à acurácia e estratégias de balanceamento.  
- É uma base **pública e amplamente utilizada em pesquisa aplicada**, sem ser uma das clássicas proibidas (Titanic, Iris, Wine etc.).

## 2. Dataset Explanation

### Contexto e Descrição
O dataset contém informações de 30.000 clientes de cartão de crédito em Taiwan.  
Cada registro representa um cliente, e a variável-alvo indica se ele **deu default no mês seguinte** (`default.payment.next.month` = 1) ou não (= 0).  
Os atributos incluem dados **demográficos**, **financeiros** e **históricos de pagamento**.

### Variáveis

**Demográficas**
- `SEX`: Gênero (1 = masculino, 2 = feminino)  
- `EDUCATION`: Grau de instrução (1 = pós-graduação, 2 = graduação, 3 = ensino médio, 4/0/5/6 = outros)  
- `MARRIAGE`: Estado civil (1 = casado, 2 = solteiro, 3/0 = outros)  
- `AGE`: Idade (anos)

**Financeiras**
- `LIMIT_BAL`: Limite total de crédito (em NT$)

**Histórico de pagamento (últimos 6 meses)**
- `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6` — Status de pagamento (valores inteiros, onde −2/−1/0 indicam pagos em dia, e 1, 2, ... indicam atraso em meses)

**Faturas mensais**
- `BILL_AMT1` a `BILL_AMT6` — Valores das faturas nos seis meses anteriores

**Pagamentos mensais**
- `PAY_AMT1` a `PAY_AMT6` — Valores pagos nos seis meses anteriores

**Alvo**
- `default.payment.next.month`: 0 = não entrou em default; 1 = entrou em default

### Tipos de dados
- **Numéricos contínuos:** `LIMIT_BAL`, `AGE`, `BILL_AMT*`, `PAY_AMT*`  
- **Numéricos discretos:** `PAY_*`  
- **Categóricos:** `SEX`, `EDUCATION`, `MARRIAGE`  
- **Binário (target):** `default.payment.next.month`

### Principais Desafios
- **Desbalanceamento:** apenas ~22% dos clientes estão em default.  
- **Categorias inválidas:** `EDUCATION` e `MARRIAGE` contêm códigos inconsistentes.  
- **Outliers:** valores muito altos em `BILL_AMT*` e `PAY_AMT*`.  
- **Multicolinearidade:** alta correlação entre séries temporais (meses consecutivos).  
- **Escalas muito diferentes:** necessidade de normalização antes do treino do MLP.

### Estatísticas e Visualizações (planejadas)
- Distribuição do alvo (`default.payment.next.month`)  
- Histogramas de `LIMIT_BAL` e `AGE`  
- Heatmap de correlação entre variáveis numéricas  
- Tabela resumo de médias, desvios e amplitudes  

### Considerações Éticas
Alguns atributos (gênero, estado civil, escolaridade) podem introduzir **viés algorítmico**.  
Discussões sobre *fairness* e mitigação de viés são pertinentes ao interpretar os resultados.

## 3. Data Cleaning and Normalization

### Estrutura e Natureza dos Dados

Os dados utilizados neste projeto representam informações de clientes de cartão de crédito, com atributos que descrevem aspectos **demográficos, financeiros e comportamentais**.  
Cada linha do dataset corresponde a um cliente, e cada coluna representa uma **feature** (atributo), como limite de crédito, idade, estado civil ou histórico de pagamento.  
Essa estrutura, em forma de **matriz de atributos**, é a base para a aplicação de técnicas de aprendizado supervisionado,  
em que cada exemplo possui um conjunto de entradas (features) e uma saída (rótulo).

As variáveis do conjunto de dados podem ser classificadas em dois tipos principais:

- **Numéricas:** representam valores contínuos ou discretos, como `LIMIT_BAL` (limite de crédito) e `AGE` (idade);  
- **Categóricas:** representam valores qualitativos, como `SEX`, `EDUCATION` e `MARRIAGE`.

Essa distinção é fundamental, pois **cada tipo de dado requer um tratamento específico** para que o modelo de aprendizado consiga interpretar corretamente as informações.

---

### Limpeza e Qualidade dos Dados

A qualidade dos dados é essencial para o desempenho de qualquer modelo de aprendizado de máquina.  
Durante a etapa de limpeza, foram verificados problemas comuns como **valores ausentes, duplicatas e inconsistências**.

- **Valores ausentes:** não foram encontrados no conjunto de dados.  
- **Duplicatas:** nenhuma linha duplicada foi identificada.  
- **Inconsistências:** categorias incorretas, como `EDUCATION = 0, 5, 6` e `MARRIAGE = 0`, foram recategorizadas como “Outros”, garantindo consistência nos dados.  
- **Valores inválidos:** apenas uma amostra foi removida por conter um valor incorreto na variável alvo (`default_payment_next_month`).

Após a limpeza, o dataset permaneceu com **30.000 amostras válidas**, todas completas e sem inconsistências estruturais.

---

### Pré-processamento e Transformação dos Dados

Como o modelo de aprendizado requer **entradas numéricas**, as variáveis categóricas foram **convertidas em formato numérico** por meio da técnica de **One-Hot Encoding**,  
criando uma coluna para cada categoria possível de `SEX`, `EDUCATION` e `MARRIAGE`.  
Esse processo garante que o modelo interprete corretamente diferenças qualitativas entre categorias, sem atribuir ordens artificiais a elas.

Em seguida, os dados numéricos foram **normalizados** para uma escala comum,  
de forma que todas as variáveis contribuam igualmente durante o treinamento da rede neural.  
Esse procedimento é essencial para evitar que atributos com valores mais altos dominem a função de custo do modelo.

Por fim, o conjunto de dados foi dividido em três subconjuntos:  
- **Treino (60%)** – usado para o aprendizado do modelo;  
- **Validação (20%)** – usado para ajuste de parâmetros;  
- **Teste (20%)** – usado para avaliar a capacidade de generalização.

A divisão foi feita de forma **estratificada**, mantendo a proporção original das classes (`default = 1` e `non-default = 0`) em todos os conjuntos.

---

### Resumo do Processo de Preparação

| Etapa | Ação Realizada |
|-------|----------------|
| Verificação de valores ausentes | Nenhum valor ausente encontrado |
| Remoção de duplicatas | Nenhuma duplicata detectada |
| Correção de categorias inválidas | Reclassificação de valores fora do intervalo válido |
| Exclusão de valores incorretos | 1 registro removido |
| Codificação de variáveis categóricas | One-Hot Encoding aplicado |
| Normalização | Escalonamento dos atributos numéricos |
| Divisão do dataset | 60% treino, 20% validação, 20% teste (estratificado) |

Esses procedimentos asseguraram que o dataset estivesse **limpo, consistente e devidamente estruturado**,  
seguindo as boas práticas de **qualidade, balanceamento e padronização de dados** recomendadas em Machine Learning.

## 4. Implementação do MLP (NumPy)

### Implementação
Implementamos um **MLP do zero**, usando apenas **NumPy** (produto matricial, ativações, softmax, cross‑entropy, backprop e atualização dos pesos). O objetivo é classificar **inadimplência** (`default_payment_next_month`) a partir dos dados já limpos/normalizados.

### Arquitetura e treino
```
Entrada (d_in) → ReLU(64) → ReLU(32) → Softmax(2)
```
- **Ativações:** ReLU nas camadas escondidas; Softmax na saída.  
- **Loss:** Cross-Entropy (com **pesos de classe** para desbalanceamento) + L2.  
- **Otimização:** **SGD mini-batch** com **momentum**, *learning rate decay* e **early stopping**.  
- **Limiar de decisão:** escolhido na validação para **máximo F1**.

### Hiperparâmetros
- Camadas escondidas: **(64, 32)**  
- Batch size: **256**  
- Épocas máx.: **60** (com *early stopping*, paciência=8)  
- Learning rate inicial: **1e‑2** (decai 0.9 a cada 5 épocas)  
- L2 (weight decay): **1e‑4**  
- Seed: **42**  
- Threshold (val, melhor F1): **0.47**

### Resultados
| Conjunto   | Acc   | Precision | Recall | F1    | ROC‑AUC |
|------------|:-----:|:---------:|:------:|:-----:|:-------:|
| **Treino** | 0.7533 | 0.4603   | 0.6660 | 0.5443 | 0.7979 |
| **Validação** | 0.7367 | 0.4325   | 0.6104 | **0.5062** | 0.7567 |
| **Teste**  | **0.7418** | **0.4425** | **0.6443** | **0.5247** | **0.7750** |

**Observação.** Em dados desbalanceados, otimizar **F1/Recall** (via threshold) pode reduzir a **accuracy** em relação ao baseline que sempre prevê a classe majoritária. Aqui priorizamos recuperar mais inadimplentes mantendo AUC e F1 sólidos.