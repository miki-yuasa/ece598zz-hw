# hw2_flow.py
# Flow Matching on three tiny datasets with full x_t trajectory recording and plotting
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
RESUME = False

EPOCHS = 200
BATCH_SIZE = 1024
LR = 2e-3

N_DATA = 20000
N_SAMPLES = 512  # number of samples for evaluation

FLOW_STEPS = 10  # flow matching steps


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
    def __init__(self, x_dim, t_dim=128, hidden=128):
        super().__init__()
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
        te = self.time_embed(t.unsqueeze(-1))
        return self.net(torch.cat([x, te], dim=-1))


# =========================
# Flow (train + sample; sample returns x_t trajectory)
# =========================
class FlowMatching:
    """
    Minimal Rectified Flow implementation
    """

    def __init__(self, model: nn.Module, device=DEVICE, x_dim=2):
        self.model = model.to(device)
        self.device = device
        self.x_dim = x_dim
        self.mse = nn.MSELoss()

    def loss(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x0: [B, x_dim]  data points
            x1: [B, x_dim]  random noise points
            t:  [B, ]       random times in (0,1)
        Returns:
            loss: scalar flow matching loss
        Function:
            L = E_{t,x0,x1} [ || v_theta( x_t, t ) - (x1 - x0) ||^2 ]
            where x_t = (1-t) x0 + t x1
        """
        # Construct the interpolated points: x_t = (1 - t) * x0 + t * x1
        t_expanded = t[:, None]  # [B, 1] for broadcasting
        xt = (1 - t_expanded) * x0 + t_expanded * x1

        # The target velocity is v_target = x1 - x0
        v_target = x1 - x0

        # Predict velocity from model
        v_pred = self.model(xt, t)

        # Compute MSE loss
        loss = self.mse(v_pred, v_target)
        return loss

    def train_one_epoch(self, loader, optimizer):
        self.model.train()
        total = 0
        for (x0,) in loader:
            x0 = x0.to(self.device)
            t = torch.rand(x0.shape[0], device=self.device)  # (0,1)
            x1 = torch.randn_like(x0)
            loss = self.loss(x0, x1, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def sample(self, n=N_SAMPLES, steps=500):
        """
        Euler integration for ODE dx/dt = v_theta(x, t)
        from t=1 → 0, returns (x0, xt_traj)
        """
        self.model.eval()
        dt = 1.0 / steps
        x = torch.randn(n, self.x_dim, device=self.device)
        traj = [x.detach().cpu().numpy()]
        for k in range(steps, 0, -1):
            # Compute t_k = k / steps
            t_k = k / steps
            t = torch.full((n,), t_k, device=self.device)

            # Evaluate velocity field at (x, t_k)
            v = self.model(x, t)

            # Euler update: x <- x - dt * v (integrating from t=1 to t=0)
            x = x - dt * v

            traj.append(x.detach().cpu().numpy())
        x0 = x.detach().cpu().numpy()
        xt_traj = np.stack(traj, axis=0)
        return x0, xt_traj


# =========================
# Checkpoint I/O (PyTorch 2.6-safe)
# =========================
def save_ckpt(path, model, optimizer, epoch, best_loss, seed=SEED):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_loss": float(best_loss),
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
    return {
        "epoch": ckpt.get("epoch", 0),
        "best_loss": ckpt.get("best_loss", float("inf")),
        "seed": ckpt.get("seed", None),
    }


# =========================
# Plotting (two functions)
# =========================
def plot_sample_paths(xt_traj, title, save_path, max_paths=200):
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
        t_vals = np.linspace(1, 0, S)
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


def plot_distributions(X, dset_name, flow_final, save_path):
    """
    1D histogram / 2D scatter
    """
    x_dim = X.shape[-1]
    fig, axes = plt.subplots(1, 2, figsize=(6, 4), sharex=True, sharey=True)
    if x_dim == 1:
        axes[0].hist(X[:N_SAMPLES, 0], bins=100, density=True, alpha=0.8)
        axes[0].set_title(f"Data ({dset_name})")
        axes[1].hist(flow_final[:, 0], bins=100, density=True, alpha=0.8)
        axes[1].set_title(f"Flow samples (steps={FLOW_STEPS})")
        for ax in axes:
            ax.set_yticks([])
    else:
        axes[0].scatter(X[:N_SAMPLES, 0], X[:N_SAMPLES, 1], s=6, alpha=0.6)
        axes[0].set_title(f"Data ({dset_name})")
        axes[1].scatter(flow_final[:, 0], flow_final[:, 1], s=6, alpha=0.6)
        axes[1].set_title(f"Flow samples (steps={FLOW_STEPS})")
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
    X: np.ndarray, flow_samples: np.ndarray, n_projections: int = 1024, seed: int = 0
):
    x_dim = X.shape[-1]
    if x_dim == 1:
        wd_flow = wasserstein_1d(X[:, 0], flow_samples[:, 0])
        print(f"[Eval] 1D Wasserstein distance:")
    else:
        wd_flow = sliced_wasserstein_2d(
            X, flow_samples, n_projections=n_projections, seed=seed
        )
        print(f"[Eval] 2D Sliced Wasserstein distance (projections={n_projections}):")
    print(f"    FLOW: {wd_flow:.6f}")


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

    # ---- model & diffusion ----
    model = MLP(x_dim=x_dim, t_dim=128, hidden=128).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    flow = FlowMatching(model=model, device=DEVICE, x_dim=x_dim)

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
        avg_loss = flow.train_one_epoch(loader, opt)
        if (ep + 1) % (EPOCHS // 5) == 0:
            print(f"[{dset_name}] epoch {ep + 1:02d}/{EPOCHS} loss={avg_loss:.6f}")
            save_ckpt(
                last_ckpt, model, opt, ep + 1, min(best_loss, avg_loss), seed=SEED
            )
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_ckpt(best_ckpt, model, opt, ep + 1, best_loss, seed=SEED)

    if os.path.isfile(best_ckpt):
        load_ckpt(best_ckpt, model, None, map_location=DEVICE)
        print(f"[{dset_name}] loaded best checkpoint.")

    # ---- sample (final distribution + trajectories) ----
    flow_final, flow_traj = flow.sample(n=N_SAMPLES, steps=FLOW_STEPS)

    # ---- plots ----
    # 1) sample paths
    plot_sample_paths(
        flow_traj,
        f"{dset_name} — FLOW (steps={FLOW_STEPS})",
        os.path.join(outdir, f"{dset_name}_paths_flow.png"),
    )

    # 2) distributions
    plot_distributions(
        X, dset_name, flow_final, os.path.join(outdir, f"{dset_name}_dist.png")
    )

    # ---- evaluation ----
    evaluate_distributions(X, flow_final, n_projections=128)


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
