# hw1.py
# DDPM & DDIM on three tiny datasets with full x_t trajectory recording and plotting
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from ot.sliced import sliced_wasserstein_distance
from scipy.stats import wasserstein_distance
from sklearn.datasets import make_s_curve
from torch.utils.data import DataLoader, TensorDataset

# =========================
# Hard-coded hyperparams
# =========================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE       = torch.device("cpu")
ROOT_OUTDIR = "runs/toys"
RESUME = True

EPOCHS = 200
BATCH_SIZE = 1024
LR = 2e-3

T = 500  # diffusion steps
BETA_START = 1e-4
BETA_END = 2e-2

N_DATA = 20000
N_SAMPLES = 2048  # number of samples to compare distributions
DDIM_STEPS = 50
ETA = 0.0  # DDIM eta (0 = deterministic)


# =========================
# Utils
# =========================
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# =========================
# Datasets
# =========================
class ToyDatasets:
    @staticmethod
    def make_scurve(n=N_DATA, seed=SEED):
        """
        2D S-curve
        """
        X3, _ = make_s_curve(n_samples=n, noise=0.03, random_state=0)
        X = X3[:, [0, 2]].astype(np.float32)
        SCALE_2D = 2.0
        X = X * SCALE_2D
        X[:, 0] *= SCALE_2D
        return X, 2, "scurve"

    @staticmethod
    def make_gmm1d2(n=N_DATA, p=0.5, mu=(-2.0, 2.0), sigma=(0.3, 0.3), seed=SEED):
        """
        1D 2 Gaussian
        """
        rng = np.random.default_rng(seed)
        z = rng.random(n) < p
        x = np.empty(n, dtype=np.float32)
        x[z] = rng.normal(mu[0], sigma[0], z.sum())
        x[~z] = rng.normal(mu[1], sigma[1], (~z).sum())
        X = x[:, None]
        X = (X - X.mean(0, keepdims=True)) / X.std(0, keepdims=True)
        return X.astype(np.float32), 1, "gmm1d2"

    @staticmethod
    def make_gmm2d8(n=N_DATA, radius=3.0, sigma=0.25, seed=SEED):
        """
        2D 8 Guassians
        """
        rng = np.random.default_rng(seed)
        k = 8
        angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
        means = np.stack(
            [radius * np.cos(angles), radius * np.sin(angles)], axis=1
        )  # [8,2]
        comp = rng.integers(0, k, size=n)
        noise = rng.normal(0.0, sigma, size=(n, 2))
        X = (means[comp] + noise).astype(np.float32)
        SCALE_2D = 2.0
        X = X * SCALE_2D
        return X, 2, "gmm2d8"


# =========================
# Model
# =========================
class MLP(nn.Module):
    def __init__(self, x_dim, t_dim=128, hidden=128, T=T):
        super().__init__()
        self.T = T
        self.time_embed = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(x_dim + t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x, t):
        t = t.float().unsqueeze(-1) / (self.T - 1)
        te = self.time_embed(t)
        return self.net(torch.cat([x, te], dim=-1))


# =========================
# Diffusion (train + sample; sample returns x_t trajectory)
# =========================
class GaussianDiffusion:
    def __init__(self, model: nn.Module, betas: torch.Tensor, device=DEVICE, x_dim=2):
        self.model = model
        self.device = device
        self.x_dim = x_dim

        self.betas = betas.to(device)
        # =========================
        # To Do
        # =========================
        self.alphas = 0
        self.alpha_bars = 0
        self.sqrt_alpha_bars = 0
        self.sqrt_one_minus_alpha_bars = 0
        self.alpha_bars_prev = 0

        self.T = betas.numel()
        self.mse = nn.MSELoss()

    # ---- training ----
    def sample_timesteps(self, bsz: int):
        return torch.randint(0, self.T, (bsz,), device=self.device)

    def q_sample(self, x0: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor):
        """
        return:
        xt:      noised data at time t
        """

        # please follow eq 4. in DDPM paper
        pass

    def p_losses(self, x0: torch.Tensor, noise: torch.Tensor):
        """
        return:
        loss: MSE loss
        """

        # please follow algo 1. in DDPM paper
        pass

    def train_one_epoch(self, loader, optimizer):
        self.model.train()
        total = 0.0
        for (x0,) in loader:
            x0 = x0.to(self.device)
            noise = torch.randn_like(x0)
            loss = self.p_losses(x0, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / len(loader)

    # ---- sampling (return final x0 and full trajectory of ALL samples) ----
    @torch.no_grad()
    def sample_ddpm(self, n=N_SAMPLES):
        """
        return:
        x0:      [n, x_dim] recon data at t=0
        xt_traj: [S, n, x_dim] samples at all time t, S = T + 1
        """

        self.model.eval()
        x = torch.randn(n, self.x_dim, device=self.device)
        traj = [x.detach().cpu().numpy()]
        for i in reversed(range(self.T)):
            # =========================
            # To Do
            # =========================
            # please follow algo 2. in DDPM paper and update x
            pass

            traj.append(x.detach().cpu().numpy())
        x0 = x.detach().cpu().numpy()
        xt_traj = np.stack(traj, axis=0)  # [T+1, n, x_dim]
        return x0, xt_traj

    @torch.no_grad()
    def sample_ddim(self, n=N_SAMPLES, steps=DDIM_STEPS, eta=ETA):
        """
        return:
        x0:      [n, x_dim] recon data at t=0
        xt_traj: [S, n, x_dim] samples at all time t, S = T + 1
        """
        self.model.eval()
        idx = np.linspace(0, self.T - 1, steps, dtype=int)
        x = torch.randn(n, self.x_dim, device=self.device)
        traj = [x.detach().cpu().numpy()]
        for i in reversed(range(steps)):
            # =========================
            # To Do
            # =========================
            # please follow eq 12 and 16 in DDPM paper and update x
            pass

            traj.append(x.detach().cpu().numpy())
        x0 = x.detach().cpu().numpy()
        xt_traj = np.stack(traj, axis=0)  # [steps+1, n, x_dim]
        return x0, xt_traj


# =========================
# Checkpoint I/O (PyTorch 2.6-safe)
# =========================
def save_ckpt(path, model, optimizer, epoch, best_loss, betas, seed=SEED):
    betas_tensor = (
        betas.detach().cpu()
        if isinstance(betas, torch.Tensor)
        else torch.as_tensor(betas, dtype=torch.float32).cpu()
    )
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "betas": betas_tensor,
        "seed": int(seed),
    }
    torch.save(ckpt, path)


def load_ckpt(path, model, optimizer=None, map_location=DEVICE):
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e1:
        print(
            f"[warn] safe load failed: {e1}\n        retrying with weights_only=False (trusted file only)"
        )
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    betas = ckpt.get("betas", None)
    if betas is not None and not isinstance(betas, torch.Tensor):
        betas = torch.as_tensor(betas, dtype=torch.float32)
    return {
        "epoch": ckpt.get("epoch", 0),
        "best_loss": ckpt.get("best_loss", float("inf")),
        "betas": betas,
        "seed": ckpt.get("seed", None),
    }


# =========================
# Plotting (two functions)
# =========================
def plot_sample_paths(xt_traj, title, save_path, T_total=T, max_paths=200):
    """
    xt_traj: [S, n, x_dim] samples at all time t, S = T + 1
    """
    S, N, D = xt_traj.shape
    M = min(N, max_paths)
    Xs = xt_traj[:, :M, :]

    line_color = "C0"
    start_color = "black"
    end_color = "C3"
    dot_size = 8

    if D == 1:
        t_vals = np.linspace(T_total - 1, 0, S)
        plt.figure(figsize=(6, 6))
        for m in range(M):
            plt.plot(
                Xs[:, m, 0],
                t_vals,
                linewidth=1.0,
                alpha=0.3,
                color=line_color,
                zorder=1,
            )
        starts_x = Xs[0, :, 0]
        ends_x = Xs[-1, :, 0]
        plt.scatter(
            starts_x,
            np.full(M, t_vals[0]),
            s=dot_size,
            alpha=0.95,
            color=start_color,
            marker="o",
            label="start",
            zorder=2,
        )
        plt.scatter(
            ends_x,
            np.full(M, t_vals[-1]),
            s=dot_size,
            alpha=0.95,
            color=end_color,
            marker="o",
            label="end",
            zorder=2,
        )
        plt.axhline(
            y=t_vals[0],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            color="gray",
            zorder=0,
        )
        plt.axhline(
            y=t_vals[-1],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            color="gray",
            zorder=0,
        )
        plt.gca().invert_yaxis()
        plt.title(f"Sample Paths — {title}")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    else:
        plt.figure(figsize=(6, 6))
        for m in range(M):
            xs = Xs[:, m, 0]
            ys = Xs[:, m, 1]
            plt.plot(xs, ys, linewidth=1.0, alpha=0.3, color=line_color, zorder=1)
        starts = Xs[0, :, :2]
        ends = Xs[-1, :, :2]
        plt.scatter(
            starts[:, 0],
            starts[:, 1],
            s=dot_size,
            alpha=0.95,
            color=start_color,
            marker="o",
            label="start",
            zorder=2,
        )
        plt.scatter(
            ends[:, 0],
            ends[:, 1],
            s=dot_size,
            alpha=0.95,
            color=end_color,
            marker="o",
            label="end",
            zorder=2,
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(f"Sample Paths — {title}")
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()


def plot_distributions(X, dset_name, ddpm_final, ddim_final, save_path):
    """
    1D histogram / 2D scatter
    """
    x_dim = X.shape[-1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    if x_dim == 1:
        axes[0].hist(X[:N_SAMPLES, 0], bins=100, density=True, alpha=0.8)
        axes[0].set_title(f"Data ({dset_name})")
        axes[1].hist(ddpm_final[:, 0], bins=100, density=True, alpha=0.8)
        axes[1].set_title("DDPM samples")
        axes[2].hist(ddim_final[:, 0], bins=100, density=True, alpha=0.8)
        axes[2].set_title(f"DDIM samples (η={ETA}, steps={DDIM_STEPS})")
        for ax in axes:
            ax.set_yticks([])
    else:
        axes[0].scatter(X[:N_SAMPLES, 0], X[:N_SAMPLES, 1], s=6, alpha=0.6)
        axes[0].set_title(f"Data ({dset_name})")
        axes[1].scatter(ddpm_final[:, 0], ddpm_final[:, 1], s=6, alpha=0.6)
        axes[1].set_title("DDPM samples")
        axes[2].scatter(ddim_final[:, 0], ddim_final[:, 1], s=6, alpha=0.6)
        axes[2].set_title(f"DDIM samples (η={ETA}, steps={DDIM_STEPS})")
        for ax in axes:
            ax.set_aspect("equal", adjustable="box")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# =========================
# Evaluation
# =========================
def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    1-Wasserstein distance
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = min(len(x), len(y))
    if len(x) != n:
        rng = np.random.default_rng(SEED)
        x = x[rng.choice(len(x), size=n, replace=False)]
    if len(y) != n:
        rng = np.random.default_rng(SEED + 1)
        y = y[rng.choice(len(y), size=n, replace=False)]
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    return float(np.mean(np.abs(x_sorted - y_sorted)))
    # return float(wasserstein_distance(x, y)) # test with scipy.stats.wasserstein_distance


def sliced_wasserstein_2d(
    X: np.ndarray, Y: np.ndarray, n_projections: int = 1024, seed: int = 0
) -> float:
    """
    sliced_wasserstein_distance from POT
    """
    return float(
        sliced_wasserstein_distance(
            X.astype(np.float64),
            Y.astype(np.float64),
            n_projections=n_projections,
            seed=seed,
        )
    )


def evaluate_distributions(
    X: np.ndarray,
    ddpm_samples: np.ndarray,
    ddim_samples: np.ndarray,
    n_projections: int = 1024,
    seed: int = 0,
):
    x_dim = X.shape[-1]
    if x_dim == 1:
        wd_ddpm = wasserstein_1d(X[:, 0], ddpm_samples[:, 0])
        wd_ddim = wasserstein_1d(X[:, 0], ddim_samples[:, 0])
        print(f"[Eval] 1D Wasserstein distance:")
        print(f"    DDPM: {wd_ddpm:.6f}")
        print(f"    DDIM: {wd_ddim:.6f}")
    else:
        wd_ddpm = sliced_wasserstein_2d(
            X, ddpm_samples, n_projections=n_projections, seed=seed
        )
        wd_ddim = sliced_wasserstein_2d(
            X, ddim_samples, n_projections=n_projections, seed=seed
        )
        print(f"[Eval] 2D Sliced Wasserstein distance (projections={n_projections}):")
        print(f"    DDPM: {wd_ddpm:.6f}")
        print(f"    DDIM: {wd_ddim:.6f}")


# =========================
# Main
# =========================
def run_one_dataset(build_fn):
    # ---- data ----
    X, x_dim, dset_name = build_fn()
    outdir = ensure_dir(os.path.join(ROOT_OUTDIR, dset_name))
    data = torch.from_numpy(X)
    loader = DataLoader(
        TensorDataset(data), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # ---- schedule ----
    betas = torch.linspace(BETA_START, BETA_END, T, dtype=torch.float32)  # linear

    # ---- model & diffusion ----
    model = MLP(x_dim=x_dim, t_dim=128, hidden=128, T=T).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    diff = GaussianDiffusion(model=model, betas=betas, device=DEVICE, x_dim=x_dim)

    # ---- resume ----
    last_ckpt = os.path.join(outdir, "last.pt")
    best_ckpt = os.path.join(outdir, "best.pt")
    start_epoch, best_loss = 0, float("inf")
    if RESUME and os.path.isfile(last_ckpt):
        extras = load_ckpt(last_ckpt, model, opt, map_location=DEVICE)
        start_epoch = int(extras.get("epoch", 0))
        best_loss = float(extras.get("best_loss", float("inf")))
        print(f"[{dset_name}] resume epoch={start_epoch}, best_loss={best_loss:.6f}")

    # ---- train ----
    for ep in range(start_epoch, EPOCHS):
        avg_loss = diff.train_one_epoch(loader, opt)
        if (ep + 1) % (EPOCHS // 5) == 0:
            print(f"[{dset_name}] epoch {ep + 1:02d}/{EPOCHS} loss={avg_loss:.6f}")
            save_ckpt(
                last_ckpt,
                model,
                opt,
                ep + 1,
                min(best_loss, avg_loss),
                betas,
                seed=SEED,
            )
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_ckpt(best_ckpt, model, opt, ep + 1, best_loss, betas, seed=SEED)

    if os.path.isfile(best_ckpt):
        load_ckpt(best_ckpt, model, None, map_location=DEVICE)
        print(f"[{dset_name}] loaded best checkpoint.")

    # ---- sample (final distribution + trajectories) ----
    ddpm_final, ddpm_traj = diff.sample_ddpm(n=N_SAMPLES)
    ddim_final, ddim_traj = diff.sample_ddim(n=N_SAMPLES, steps=DDIM_STEPS, eta=ETA)

    # ---- plots ----
    # 1) sample paths
    plot_sample_paths(
        ddpm_traj,
        f"{dset_name} — DDPM",
        os.path.join(outdir, f"{dset_name}_paths_ddpm.png"),
        T_total=T,
    )
    plot_sample_paths(
        ddim_traj,
        f"{dset_name} — DDIM (η={ETA}, steps={DDIM_STEPS})",
        os.path.join(outdir, f"{dset_name}_paths_ddim.png"),
        T_total=T,
    )

    # 2) distributions
    plot_distributions(
        X,
        dset_name,
        ddpm_final,
        ddim_final,
        os.path.join(outdir, f"{dset_name}_dist.png"),
    )

    # ---- evaluation ----
    evaluate_distributions(X, ddpm_final, ddim_final)


def main():
    seed_everything(SEED)
    ensure_dir(ROOT_OUTDIR)

    # 3 datasets
    run_one_dataset(ToyDatasets.make_gmm1d2)
    run_one_dataset(ToyDatasets.make_gmm2d8)
    run_one_dataset(ToyDatasets.make_scurve)

    print("[done] all datasets processed. See:", ROOT_OUTDIR)


if __name__ == "__main__":
    main()
