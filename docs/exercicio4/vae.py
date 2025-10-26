# vae_train.py
# ---------------------------------------------------------
# Treinamento completo de um VAE (ConvVAE) em MNIST/Fashion-MNIST:
# - Data preparation (load/normalize/split)
# - Model implementation (encoder, decoder, reparameterization)
# - Training loop (ELBO = Recon + beta*KL)
# - Evaluation em validação
# - Visualizações: reconstruções, samples, espaço latente (2D/3D ou PCA/t-SNE)
# - Saídas: PNGs, history.csv, best.pt e um mini relatório Markdown
# ---------------------------------------------------------
import argparse
import math
import os
import time
from datetime import datetime
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms, utils as vutils

# Visualizações
import matplotlib

matplotlib.use("Agg")  # para salvar figuras sem janela
import matplotlib.pyplot as plt

# Redução de dimensionalidade (fallback se z>3)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# =========================
# Utilidades gerais
# =========================
def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
# Modelo VAE (Conv)
# =========================
class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder para imagens 1x28x28 (MNIST/Fashion-MNIST).
    Encoder retorna (mu, logvar). Decoder mapeia z -> logits (antes do sigmoid).
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x28x28 -> 32x14x14 -> 64x7x7 -> 128x7x7
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc_out_dim = 128 * 7 * 7
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1),  # logits
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.size(0), 128, 7, 7)
        logits = self.decoder(h)
        return logits

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


# =========================
# Perda (ELBO)
# =========================
def vae_loss(logits, x, mu, logvar, beta=1.0, recon_loss="bce"):
    """
    logits: saída do decoder (antes de sigmoid)
    x: alvo em [0,1]
    recon_loss: 'bce' (with logits) ou 'mse'
    retorna: elbo (média por item), recon (média por item), kl (média por item)
    """
    b = x.size(0)
    if recon_loss == "bce":
        recon_sum = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
    elif recon_loss == "mse":
        recon_sum = F.mse_loss(torch.sigmoid(logits), x, reduction="sum")
    else:
        raise ValueError("recon_loss must be 'bce' or 'mse'")

    kl_sum = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elbo_sum = recon_sum + beta * kl_sum

    # também retornamos por elemento (médio por item no batch)
    return elbo_sum / b, recon_sum / b, kl_sum / b


# =========================
# Dados
# =========================
def get_dataloaders(dataset_name, data_dir, batch_size, val_split, seed):
    tfm = transforms.ToTensor()  # normaliza para [0,1]
    if dataset_name == "mnist":
        full = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
        test = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)
        num_classes = 10
    elif dataset_name == "fashion":
        full = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=tfm
        )
        test = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=tfm
        )
        num_classes = 10
    else:
        raise ValueError("dataset must be 'mnist' or 'fashion'")

    n_total = len(full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return (
        train_loader,
        val_loader,
        test_loader,
        (1, 28, 28),
        n_train,
        n_val,
        len(test),
        num_classes,
    )


# =========================
# Visualizações & salvamento
# =========================
@torch.no_grad()
def save_reconstructions(model, loader, device, outdir, tag="val", n=16):
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device)[:n]
    logits, _, _ = model(x)
    x_hat = torch.sigmoid(logits)

    # grid com originais em cima e reconstruções embaixo
    comp = torch.cat([x, x_hat], dim=0)
    grid = vutils.make_grid(comp, nrow=n, padding=2)
    path = os.path.join(outdir, f"reconstructions_{tag}.png")
    vutils.save_image(grid, path)
    return path


@torch.no_grad()
def save_samples(model, device, outdir, tag="val", n=64):
    model.eval()
    z = torch.randn(n, model.latent_dim, device=device)
    logits = model.decode(z)
    samples = torch.sigmoid(logits)
    grid = vutils.make_grid(samples, nrow=int(math.sqrt(n)))
    path = os.path.join(outdir, f"samples_{tag}.png")
    vutils.save_image(grid, path)
    return path


@torch.no_grad()
def save_latent_plot(
    model, loader, device, outdir, latent_dim, tag="val", method="auto"
):
    """
    Salva:
      - scatter 2D/3D de mu, se latent_dim<=3
      - caso contrário, redução via PCA (2D) ou t-SNE (2D) se method='tsne'
    """
    model.eval()
    mus = []
    ys = []
    for x, y in loader:
        x = x.to(device)
        mu, logvar = model.encode(x)
        mus.append(mu.detach().cpu())
        ys.append(y)
    Z = torch.cat(mus, dim=0).numpy()
    Y = torch.cat(ys, dim=0).numpy()

    if latent_dim == 1:
        fig = plt.figure(figsize=(6, 4))
        plt.scatter(Z[:, 0], [0] * len(Z), c=Y, s=6, alpha=0.7, cmap="tab10")
        plt.xlabel("z1")
        plt.yticks([])
        plt.title(f"Latent (mu) — {tag}")
        path = os.path.join(outdir, f"latent_{tag}_1d.png")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close(fig)
        return path

    if latent_dim == 2:
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(Z[:, 0], Z[:, 1], c=Y, s=6, alpha=0.7, cmap="tab10")
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.title(f"Latent (mu) — {tag}")
        path = os.path.join(outdir, f"latent_{tag}_2d.png")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close(fig)
        return path

    if latent_dim == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        p = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=Y, s=6, alpha=0.7, cmap="tab10")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_zlabel("z3")
        fig.colorbar(p, shrink=0.6)
        ax.set_title(f"Latent (mu) — {tag}")
        path = os.path.join(outdir, f"latent_{tag}_3d.png")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close(fig)
        return path

    # latent_dim > 3: reduzir
    if method == "tsne":
        Z2 = TSNE(
            n_components=2, init="pca", learning_rate="auto", perplexity=30
        ).fit_transform(Z)
        tag2 = "tsne2d"
    else:
        Z2 = PCA(n_components=2).fit_transform(Z)
        tag2 = "pca2d"

    fig = plt.figure(figsize=(6, 5))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=Y, s=6, alpha=0.7, cmap="tab10")
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.title(f"Latent (mu) — {tag} ({tag2})")
    path = os.path.join(outdir, f"latent_{tag}_{tag2}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_history_csv(history: List[Dict], outdir: str):
    import csv

    if not history:
        return None
    path = os.path.join(outdir, "history.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    return path


def write_mini_report(
    outdir: str, args, best_epoch: int, best_val_elbo: float, artifacts: Dict[str, str]
):
    report_path = os.path.join(outdir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# VAE Report\n\n")
        f.write(f"**Dataset:** {args.dataset}\n\n")
        f.write(f"**Latent dim:** {args.latent_dim}\n\n")
        f.write(f"**β (beta-VAE):** {args.beta}\n\n")
        f.write(f"**Recon loss:** {args.recon_loss}\n\n")
        f.write(f"**Epochs:** {args.epochs}\n\n")
        f.write(f"**Batch size:** {args.batch_size}\n\n")
        f.write(f"**LR:** {args.lr}\n\n")
        f.write(f"**Melhor época (val ELBO):** {best_epoch}\n\n")
        f.write(f"**Melhor val_ELBO:** {best_val_elbo:.6f}\n\n")
        f.write("## Artefatos gerados\n")
        for name, path in artifacts.items():
            if path:
                f.write(f"- {name}: `{path}`\n")
        f.write("\n## Observações & Insights\n")
        f.write(
            "- Reconstruções permitem comparar qualidade visual vs. regularização (KL).\n"
        )
        f.write(
            "- `β` maior tende a organizar melhor o espaço latente (interpolação suave), às vezes piorando a nitidez das reconstruções.\n"
        )
        f.write(
            "- Se `latent_dim<=3`, a visualização direta de `μ` revela separabilidade por classes; com `latent_dim` alto, PCA/t-SNE são úteis.\n"
        )
        f.write(
            "\n## Próximos passos\n- Grid de hiperparâmetros (β, `latent_dim`).\n- Interpolação no espaço latente.\n- Métrica FID (opcional) para comparar qualidade amostral.\n"
        )
    return report_path


# =========================
# Treino / Validação
# =========================
def train_one_epoch(model, loader, opt, device, beta, recon_loss):
    model.train()
    running = {"elbo": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
    t0 = time.time()
    for x, _ in loader:
        x = x.to(device)
        opt.zero_grad()
        logits, mu, logvar = model(x)
        elbo, recon, kl = vae_loss(
            logits, x, mu, logvar, beta=beta, recon_loss=recon_loss
        )
        elbo.backward()
        opt.step()
        bs = x.size(0)
        running["elbo"] += elbo.item() * bs
        running["recon"] += recon.item() * bs
        running["kl"] += kl.item() * bs
        running["n"] += bs
    dt = time.time() - t0
    for k in ["elbo", "recon", "kl"]:
        running[k] /= max(1, running["n"])
    running["sec_per_epoch"] = dt
    return running


@torch.no_grad()
def evaluate(model, loader, device, beta, recon_loss):
    model.eval()
    running = {"elbo": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
    for x, _ in loader:
        x = x.to(device)
        logits, mu, logvar = model(x)
        elbo, recon, kl = vae_loss(
            logits, x, mu, logvar, beta=beta, recon_loss=recon_loss
        )
        bs = x.size(0)
        running["elbo"] += elbo.item() * bs
        running["recon"] += recon.item() * bs
        running["kl"] += kl.item() * bs
        running["n"] += bs
    for k in ["elbo", "recon", "kl"]:
        running[k] /= max(1, running["n"])
    return running


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="VAE Training — MNIST/Fashion-MNIST")
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--recon-loss", choices=["bce", "mse"], default="bce")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="./vae_runs")
    parser.add_argument(
        "--latent-reduction",
        choices=["auto", "pca", "tsne"],
        default="auto",
        help="Se latent_dim>3: escolha redução p/ visualização.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pastas
    run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(
        args.outdir, f"{args.dataset}_z{args.latent_dim}_beta{args.beta}_{run_tag}"
    )
    ensure_dir(outdir)

    print("=== Configuração ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"Device: {device}")
    print(f"Saídas em: {outdir}")

    # Dados
    train_loader, val_loader, test_loader, in_shape, n_train, n_val, n_test, _ = (
        get_dataloaders(
            args.dataset, args.data_dir, args.batch_size, args.val_split, args.seed
        )
    )
    print("\n=== Dados ===")
    print(f"Entrada (C,H,W): {in_shape}")
    print(f"Treino: {n_train} | Validação: {n_val} | Teste: {n_test}")

    # Modelo
    model = ConvVAE(in_channels=1, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = count_parameters(model)
    print("\n=== Modelo ===")
    print(model)
    print(f"Parâmetros treináveis: {n_params:,}")

    # Artefatos iniciais
    print("\n=== Amostras iniciais (prior) ===")
    p_s0 = save_samples(model, device, outdir, tag="epoch0", n=64)
    print(f"Samples (z~N(0,I)) salvos em: {p_s0}")

    # Treino
    print("\n=== Treinando ===")
    best_val_elbo, best_epoch = float("inf"), -1
    history = []
    H, W = in_shape[1], in_shape[2]

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model, train_loader, opt, device, beta=args.beta, recon_loss=args.recon_loss
        )
        va = evaluate(
            model, val_loader, device, beta=args.beta, recon_loss=args.recon_loss
        )

        elbo_px_tr = tr["elbo"] / (H * W)
        elbo_px_va = va["elbo"] / (H * W)
        print(
            f"[Epoch {epoch:03d}] "
            f"train_ELBO={tr['elbo']:.4f} (px={elbo_px_tr:.6f})  "
            f"val_ELBO={va['elbo']:.4f} (px={elbo_px_va:.6f})  |  "
            f"train_recon={tr['recon']:.4f}  val_recon={va['recon']:.4f}  |  "
            f"train_KL={tr['kl']:.4f}  val_KL={va['kl']:.4f}  |  "
            f"time={tr['sec_per_epoch']:.2f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_elbo": tr["elbo"],
                "val_elbo": va["elbo"],
                "train_recon": tr["recon"],
                "val_recon": va["recon"],
                "train_kl": tr["kl"],
                "val_kl": va["kl"],
                "elbo_px_train": elbo_px_tr,
                "elbo_px_val": elbo_px_va,
                "epoch_time_sec": tr["sec_per_epoch"],
            }
        )

        # Visualizações periódicas
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            p_samples = save_samples(model, device, outdir, tag=f"epoch{epoch}", n=64)
            p_recon = save_reconstructions(
                model, val_loader, device, outdir, tag=f"epoch{epoch}", n=16
            )
            print(f"  -> Samples: {p_samples}")
            print(f"  -> Reconstruções: {p_recon}")
            # Latent plot (mu)
            tag = f"epoch{epoch}"
            if args.latent_dim <= 3:
                p_lat = save_latent_plot(
                    model, val_loader, device, outdir, args.latent_dim, tag=tag
                )
            else:
                method = {"auto": "pca", "pca": "pca", "tsne": "tsne"}[
                    args.latent_reduction
                ]
                p_lat = save_latent_plot(
                    model,
                    val_loader,
                    device,
                    outdir,
                    args.latent_dim,
                    tag=tag,
                    method=method,
                )
            print(f"  -> Latent plot: {p_lat}")

        # Checkpoint do melhor
        if va["elbo"] < best_val_elbo:
            best_val_elbo = va["elbo"]
            best_epoch = epoch
            ckpt_path = os.path.join(outdir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "args": vars(args),
                    "val_elbo": best_val_elbo,
                },
                ckpt_path,
            )
            print(
                f"  ** Novo melhor (val_ELBO={best_val_elbo:.4f}) no epoch {epoch}. Checkpoint salvo em {ckpt_path}"
            )

    # Avaliação final (val e teste)
    print("\n=== Avaliação final ===")
    va = evaluate(model, val_loader, device, beta=args.beta, recon_loss=args.recon_loss)
    te = evaluate(
        model, test_loader, device, beta=args.beta, recon_loss=args.recon_loss
    )
    print(f"Val:  ELBO={va['elbo']:.6f}  Recon={va['recon']:.6f}  KL={va['kl']:.6f}")
    print(f"Test: ELBO={te['elbo']:.6f}  Recon={te['recon']:.6f}  KL={te['kl']:.6f}")

    # Visualizações finais
    p_samples_final = save_samples(model, device, outdir, tag="final", n=64)
    p_recon_final = save_reconstructions(
        model, test_loader, device, outdir, tag="final", n=16
    )
    if args.latent_dim <= 3:
        p_lat_final = save_latent_plot(
            model, test_loader, device, outdir, args.latent_dim, tag="final"
        )
    else:
        method = {"auto": "pca", "pca": "pca", "tsne": "tsne"}[args.latent_reduction]
        p_lat_final = save_latent_plot(
            model,
            test_loader,
            device,
            outdir,
            args.latent_dim,
            tag="final",
            method=method,
        )

    # Histórico e mini relatório
    hist_csv = save_history_csv(history, outdir)
    artifacts = {
        "history_csv": hist_csv,
        "samples_initial": p_s0,
        "samples_final": p_samples_final,
        "reconstructions_final": p_recon_final,
        "latent_plot_final": p_lat_final,
    }
    report_md = write_mini_report(outdir, args, best_epoch, best_val_elbo, artifacts)

    print("\n=== Concluído ===")
    print(f"Melhor época (val ELBO): {best_epoch}")
    print(f"Melhor val ELBO: {best_val_elbo:.6f}")
    print(f"Artefatos: {artifacts}")
    print(f"Relatório: {report_md}")
    print(f"Saídas em: {outdir}")


if __name__ == "__main__":
    main()
