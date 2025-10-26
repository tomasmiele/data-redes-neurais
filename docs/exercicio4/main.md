# Exercício — VAE

## Setup

```bash
# Ambiente (venv "env")
python -m pip uninstall -y torch torchvision torchaudio numpy
python -m pip install "numpy<2" --upgrade
python -m pip install "torch==2.2.2" "torchvision==0.17.2"
python -m pip install matplotlib scikit-learn pillow

# Execução
python vae.py --dataset mnist --epochs 10 --latent-dim 2 --beta 1.0
```

**Configuração usada**
```
dataset: mnist
epochs: 10
batch_size: 128
latent_dim: 2
beta: 1.0
lr: 0.001
recon_loss: bce
val_split: 0.1
seed: 42
device: cpu
```

**Observação de download:** os links originais de MNIST (yann.lecun.com) deram 404; o `torchvision` baixou automaticamente de `ossci-datasets.s3.amazonaws.com`.

---

## Data Preparation

- **Load** MNIST (train/test) via `torchvision.datasets.MNIST`.
- **Normalize** em \([0,1]\) com `transforms.ToTensor()`.
- **Split** treino/val com `random_split` (90%/10%):  
  **Treino:** 54.000 | **Validação:** 6.000 | **Teste:** 10.000

---

## Model Implementation

### Arquitetura (ConvVAE, 1×28×28)
**Encoder**
- Conv(1→32, k4,s2,p1) → ReLU ⟶ 14×14  
- Conv(32→64, k4,s2,p1) → ReLU ⟶ 7×7  
- Conv(64→128, k3,s1,p1) → ReLU ⟶ 7×7  
- Flatten(128·7·7=6272) → FC_μ (6272→2), FC_{logvar} (6272→2)

**Decoder**
- FC(2→6272) → reshape (128×7×7)  
- ConvT(128→64, k4,s2,p1) → ReLU ⟶ 14×14  
- ConvT(64→32, k4,s2,p1) → ReLU ⟶ 28×28  
- Conv(32→1, k3,s1,p1) ⟶ **logits**

**Reparameterization trick**
\[
z = \mu + \sigma \odot \varepsilon,\;\varepsilon\sim\mathcal{N}(0,I),\;\sigma=\exp\!\big(0.5\cdot \log\sigma^2\big)
\]

**Parâmetros treináveis:** **315.365**

---

## Loss (ELBO)

- **Reconstrução (BCE with logits, sum)** + **β · KL** (com β=1.0).  
- KL (gaussianas diagonais):  
  \[
  -\tfrac12 \sum_i \left(1 + \log\sigma_i^2 - \mu_i^2 - \sigma_i^2\right)
  \]
- Métricas reportadas **por item** no batch; também mostramos **ELBO por pixel**.

---

## Training

**Amostras iniciais (prior)**: `samples_epoch0.png`  
Treino por 10 épocas com Adam (lr=1e-3), batch=128.

| Epoch | train_ELBO | val_ELBO | train_Recon | val_Recon | train_KL | val_KL | ELBO/pixel (val) |
|------:|-----------:|---------:|------------:|----------:|---------:|-------:|------------------:|
| 1 | 189.9199 | **166.6079** | 185.5278 | 161.7278 | 4.3921 | 4.8801 | 0.212510 |
| 2 | 161.6840 | **157.6834** | 156.4333 | 152.2417 | 5.2507 | 5.4416 | 0.201127 |
| 3 | 156.0742 | **154.2295** | 150.4499 | 148.3952 | 5.6243 | 5.8342 | 0.196721 |
| 4 | 153.3052 | **152.5510** | 147.4894 | 146.6769 | 5.8157 | 5.8741 | 0.194580 |
| 5 | 151.5527 | **151.5376** | 145.6390 | 145.1763 | 5.9136 | 6.3614 | 0.193288 |
| 6 | 150.1056 | **149.6207** | 144.0820 | 143.3907 | 6.0236 | 6.2300 | 0.190843 |
| 7 | 149.1560 | **149.2737** | 143.0495 | 142.9689 | 6.1065 | 6.3048 | 0.190400 |
| 8 | 148.1687 | **149.1837** | 142.0027 | 142.9922 | 6.1660 | 6.1915 | 0.190285 |
| 9 | 147.5587 | **148.2316** | 141.3345 | 142.0705 | 6.2242 | 6.1611 | 0.189071 |
| 10| 146.9479 | **148.2099** | 140.6753 | 141.8223 | 6.2727 | 6.3876 | 0.189043 |

**Melhor época (val ELBO):** **10** (val_ELBO ≈ **148.210**).  
**Checkpoints:** `best.pt` salvo quando melhora.

**Reconstruções e latente durante treino:**  
- `reconstructions_epoch1.png`, `latent_epoch1_2d.png`  
- `reconstructions_epoch5.png`, `latent_epoch5_2d.png`  
- `reconstructions_epoch10.png`, `latent_epoch10_2d.png`

---

## Evaluation

**Validação (final):**  
ELBO = **148.274** • Recon = **141.886** • KL = **6.388**

**Teste (final):**  
ELBO = **147.870** • Recon = **141.483** • KL = **6.387**

**Amostras do prior (final):** `samples_final.png`  
**Reconstruções (final):** `reconstructions_final.png`  
**Espaço latente (μ, 2D):** `latent_final_2d.png`

---

## Visualization

- **Originais vs Reconstruções:** grids salvos por época e no final.
- **Espaço latente (z=2):** *scatter* colorido por rótulo — separabilidade crescente ao longo do treino.
- **Amostras do prior:** grades de dígitos gerados a partir de \(z\sim\mathcal{N}(0,I)\).

---

## Análise e Resultados

- **Convergência estável:** queda consistente do **val ELBO** até ~epoch 10; o termo **Recon** domina a ELBO (~142) com **KL** (~6–6.4), típico de VAE com β=1.  
- **Latente 2D significativo:** visualização mostra clusters por classe; qualidade melhora entre as épocas 1→5→10.  
- **Reconstruções razoáveis:** nitidez compatível com MNIST, com leve borramento inerente ao VAE (BCE+logits).  
- **Trade-off Recon↔KL:** KL cresce sutilmente (6.0→6.4), organizando o espaço latente sem degradar demais a Recon.

**Desafios enfrentados**
- **Compatibilidade NumPy/PyTorch:** necessidade de usar `numpy<2` com as rodas de PyTorch/Torchvision disponíveis.  
- **Download de MNIST:** links originais retornaram 404; fallback automático funcionou.  
- **CPU-only:** tempos por época maiores (≈ 44–79s); ainda assim, convergência adequada em 10 épocas.

**Insights**
- **β=1.0** já produz um latente 2D útil; aumentar β tende a estruturar ainda mais o espaço (com possível perda de nitidez).  
- **Latent dim=2** facilita inspeção visual e interpolação; para **z>3**, usar PCA/t-SNE ajuda a manter a interpretabilidade.

---

## Próximos Passos

- **β-sweep** \(\beta\in\{0.5,1,2,4\}\) e **z-sweep** \(z\in\{2,8,16,32\}\) para estudar Recon↔KL.  
- **Interpolações** no espaço latente entre pares de dígitos.  
- **Métrica FID** (opcional) para qualidade amostral.  
- **Augmentations leves** (ex.: shifts) e **scheduler** de LR para refino.

---

## Artefatos Gerados

- `history.csv` — métricas por época  
- `best.pt` — melhor checkpoint  
- `samples_epoch0.png`, `samples_epoch5.png`, `samples_epoch10.png`, `samples_final.png`  
- `reconstructions_epoch1.png`, `reconstructions_epoch5.png`, `reconstructions_epoch10.png`, `reconstructions_final.png`  
- `latent_epoch1_2d.png`, `latent_epoch5_2d.png`, `latent_epoch10_2d.png`, `latent_final_2d.png`  
- `report.md` — mini‐relatório automático do script

---
