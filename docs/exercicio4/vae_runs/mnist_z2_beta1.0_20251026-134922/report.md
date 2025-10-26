# VAE Report

**Dataset:** mnist

**Latent dim:** 2

**β (beta-VAE):** 1.0

**Recon loss:** bce

**Epochs:** 10

**Batch size:** 128

**LR:** 0.001

**Melhor época (val ELBO):** 10

**Melhor val_ELBO:** 148.209887

## Artefatos gerados
- history_csv: `./vae_runs/mnist_z2_beta1.0_20251026-134922/history.csv`
- samples_initial: `./vae_runs/mnist_z2_beta1.0_20251026-134922/samples_epoch0.png`
- samples_final: `./vae_runs/mnist_z2_beta1.0_20251026-134922/samples_final.png`
- reconstructions_final: `./vae_runs/mnist_z2_beta1.0_20251026-134922/reconstructions_final.png`
- latent_plot_final: `./vae_runs/mnist_z2_beta1.0_20251026-134922/latent_final_2d.png`

## Observações & Insights
- Reconstruções permitem comparar qualidade visual vs. regularização (KL).
- `β` maior tende a organizar melhor o espaço latente (interpolação suave), às vezes piorando a nitidez das reconstruções.
- Se `latent_dim<=3`, a visualização direta de `μ` revela separabilidade por classes; com `latent_dim` alto, PCA/t-SNE são úteis.

## Próximos passos
- Grid de hiperparâmetros (β, `latent_dim`).
- Interpolação no espaço latente.
- Métrica FID (opcional) para comparar qualidade amostral.
