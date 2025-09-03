import pandas as pd

df = pd.read_csv("docs/exercicio1/spaceship-titanic/train.csv")

missing_counts = df.isnull().sum()
missing = missing_counts[missing_counts > 0].sort_values(ascending=False)

print(f"Colunas com valores faltantes:\n{missing}")

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def split_passenger_id(s):
    # Ex.: "0001_01" -> group=1, number=1
    try:
        g, n = s.split("_")
        return int(g), int(n)
    except Exception:
        return np.nan, np.nan

def split_cabin(s):
    # Ex.: "B/45/P" -> deck='B', num=45, side='P'
    try:
        deck, num, side = s.split("/")
        num = pd.to_numeric(num, errors="coerce")
        return deck, num, side
    except Exception:
        return np.nan, np.nan, np.nan

# ------------- 4) Feature engineering leve -------------
# Decompor PassengerId e Cabin; manteremos as partes úteis e descartaremos os originais
df[["GroupId", "PassengerNum"]] = df["PassengerId"].apply(lambda s: pd.Series(split_passenger_id(s)))
df[["Deck", "CabinNum", "Side"]] = df["Cabin"].apply(lambda s: pd.Series(split_cabin(s)))

# Variáveis que não vão para a rede:
drop_cols = ["PassengerId", "Name", "Cabin"]  # Name é texto livre; PassengerId/Cabin já decompostos
for c in drop_cols:
    if c in df.columns:
        df.drop(columns=c, inplace=True)

# ------------- 5) Definição de colunas por tipo -------------
target_col = "Transported"
num_cols = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "PassengerNum", "GroupId", "CabinNum"
]
cat_cols = [
    "HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"
]

# ------------- 6) Estratégias de imputação -------------
# Justificativas:
# - Numéricas (gastos e idade) são normalmente assimétricas → usar MEDIANA é robusto a outliers.
# - Categóricas → 'most_frequent' mantém distribuição das categorias.
# - Booleans (CryoSleep, VIP) também tratadas como categóricas; depois podem virar 0/1 com OneHot.

numeric_imputer = SimpleImputer(strategy="median")
categorical_imputer = SimpleImputer(strategy="most_frequent")

# ------------- 7) Encoding + Scaling -------------
# Escalonador: normaliza features numéricas para [-1, 1],
# o que é ideal para tanh (centrado em 0, evita saturação nas extremidades)
scaler = MinMaxScaler(feature_range=(-1, 1))
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

numeric_pipeline = Pipeline(steps=[
    ("imputer", numeric_imputer),
    ("scaler", scaler),
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", categorical_imputer),
    ("onehot", ohe),
])

pre = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ],
    remainder="drop"
)

# ------------- 8) Aplicar somente em X; manter alvo separadamente -------------
X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
y = None
if target_col in df.columns:
    # Convertê-la para 0/1 (True->1, False->0)
    y = df[target_col].astype("int64")

X_proc = pre.fit_transform(X)

num_names = num_cols
cat_names = []
if len(cat_cols) > 0:
    ohe_feat_names = pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols)
    cat_names = ohe_feat_names.tolist()
feature_names = num_names + cat_names

X_proc_df = pd.DataFrame(X_proc, columns=feature_names, index=X.index)
if y is not None:
    X_proc_df[target_col] = y.values

X_proc_df.to_csv("docs/exercicio1/spaceship-titanic/train_preprocessed.csv", index=False)

import matplotlib.pyplot as plt

# --------- FoodCourt (numérica) ---------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Original
df["FoodCourt"].hist(ax=axes[0], bins=30, color="skyblue", edgecolor="black")
axes[0].set_title("FoodCourt (antes do scaling)")
axes[0].set_xlabel("Valor original")
axes[0].set_ylabel("Frequência")

# Pós-preprocessamento (pegar da matriz transformada)
foodcourt_idx = feature_names.index("FoodCourt")
pd.Series(X_proc[:, foodcourt_idx]).hist(ax=axes[1], bins=30, color="lightgreen", edgecolor="black")
axes[1].set_title("FoodCourt (após MinMaxScaler [-1,1])")
axes[1].set_xlabel("Valor escalado")
axes[1].set_ylabel("Frequência")

plt.tight_layout()
plt.show()

# --------- HomePlanet (categórica) ---------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Original
df["HomePlanet"].value_counts().plot(kind="bar", ax=axes[0], color="coral", edgecolor="black")
axes[0].set_title("HomePlanet (antes do One-Hot)")
axes[0].set_xlabel("Planeta")
axes[0].set_ylabel("Contagem")

# Pós-OneHot (cada planeta vira coluna binária)
homeplanet_cols = [c for c in feature_names if c.startswith("HomePlanet_")]
pd.DataFrame(X_proc, columns=feature_names)[homeplanet_cols].sum().plot(kind="bar", ax=axes[1], color="lightseagreen", edgecolor="black")
axes[1].set_title("HomePlanet (após One-Hot Encoding)")
axes[1].set_xlabel("Colunas geradas")
axes[1].set_ylabel("Contagem de 1s")

plt.tight_layout()
plt.show()
