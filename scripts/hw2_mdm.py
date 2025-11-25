# hw2_discrete.py
# Discrete Diffusion on three tiny datasets with full x_t trajectory recording and plotting
import os, random
import math
import argparse
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange


# =========================
# Hard-coded hyperparams
# =========================
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE       = torch.device("cpu")
ROOT_OUTDIR  = "runs/toys"
RESUME       = True
SAVE_EVERY   = 200
TOTAL_STEPS  = 1000
BATCH_SIZE   = 1024
LR           = 2e-3

N_SAMPLES    = 512         # number of samples for evaluation


# =========================
# Utils
# =========================
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

# =========================
# Datasets
# =========================
class ToyDatasets:
    def __init__(self, name):
        if name == 'cont_pairs4':
            # e.g. [0,1,2,3], [3,4,5,6], [8,9,0,1]
            self.name = name
            self.vocab = V = 10
            self.seq_len = 4
            self.data = [[i, (i+1) % V, (i+2) % V, (i+3) % V] for i in range(V)]
            self.dataset_len = len(self.data)
            self.P = np.zeros((V, V, V, V), dtype=float)
            for i in range(V):
                self.P[i, (i+1)%V, (i+2)%V, (i+3)%V] = 1/V
        else:
            raise NotImplementedError

    def make_batch(self, batch_size: int, device) -> torch.Tensor:
        idx = torch.randint(low=0, high=self.dataset_len, size=(batch_size,), device=device)
        batch = torch.tensor(self.data, device=device, dtype=torch.long)[idx]
        return batch

# =========================
# Model
# =========================
class TinyTransformer(nn.Module):
    def __init__(self, vocab: int, seq_len: int, d_model: int = 32, nhead: int = 4, nlayers: int = 2, t_dim: int = 16):
        super().__init__()
        self.vocab = vocab
        self.seq_len = seq_len
        self.t_dim = t_dim

        self.token_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)

        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.t_proj = nn.Linear(t_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, xt: torch.Tensor, t: torch.Tensor):
        B = xt.size(0)
        x = self.token_emb(xt)  # [B,L,D]

        t_emb = sinusoidal_t_embed(t, dim=self.t_dim)            # [B,T]
        t_add = self.t_proj(t_emb)[:, None, :].expand(B, self.seq_len, -1)
        x = x + self.pos_emb + t_add

        h = self.enc(x)
        h = self.norm(h)
        logits = self.head(h)  # [B,L,V]
        return logits

# =========================
# Diffusion (train + sample; sample returns x_t trajectory)
# =========================
class MDM:
    """
    Minimal Masked Diffusion Model implementation
    """
    def __init__(self, model: nn.Module, vocab: int, device=DEVICE):
        self.model = model
        self.device = device
        self.V = vocab + 1       # data + mask_id
        self.mask_id = vocab

    def qt_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x0: (B, L) long tensor, input token sequence.
            t: (B, ) float tensor in [0,1], corruption level (t=0 → data, t=1 → fully masked).
        Returns:
            mask: (B, L) bool, True iff that position is sampled to be masked.
            xt: (B, L) long, masked sequence with mask_id.
        Function:
            Samples q(xt | x0) = Cat( (1-t) x0 + t \delta_{[MASK]})
        """
        B, L = x0.shape
        # Sample uniform random values for each position
        rand = torch.rand(B, L, device=x0.device)
        
        # Mask with probability t for each example
        # t is (B,), expand to (B, L) for comparison
        t_expanded = t[:, None].expand(B, L)
        
        # mask is True where we mask (with probability t)
        mask = rand < t_expanded
        
        # Create corrupted sequence
        xt = x0.clone()
        xt[mask] = self.mask_id
        
        return mask, xt

    def loss(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x0: (B, L) long tensor, input token sequence.
            t: (B, ) float tensor in [0,1], corruption level (t=0 → data, t=1 → fully masked).
        Returns:
            loss: scalar tensor, the loss value.
        Function:
            Computes the loss L = - E_{t,x0,xt} [ 1/t \Sum_i 1{x_t^i=M} log p(x_0^i | x_t) ]
            Eq. (3) in LLaDA paper.
        """
        B, L = x0.shape
        
        # Sample corrupted sequence
        mask, xt = self.qt_sample(x0, t)
        
        # Get model predictions: logits over vocabulary for each position
        logits = self.model(xt, t)  # (B, L, V)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)
        
        # Get log probability of the true token x0 at each position
        # log_probs is (B, L, V), x0 is (B, L)
        log_probs_true = log_probs.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L)
        
        # Mask out unmasked positions (only compute loss for masked positions)
        # mask is (B, L) boolean
        log_probs_masked = log_probs_true * mask.float()  # (B, L)
        
        # Sum over positions for each example
        sum_log_probs = log_probs_masked.sum(dim=1)  # (B,)
        
        # Weight by 1/t per example and negate
        weighted_loss = -sum_log_probs / t  # (B,)
        
        # Average over batch
        loss = weighted_loss.mean()
        
        return loss

    def train_one_step(self, x0, optimizer):
        self.model.train()
        t = torch.rand(x0.shape[0], device=self.device) * 0.6 + 0.2
        loss = self.loss(x0, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    @torch.no_grad()
    def sample(self, n: int, T: int = 4, temperature: float = 1.0) -> torch.Tensor:
        """
        Args:
            n: int, number of samples to generate.
            T: int, number of diffusion steps.
            temperature: float, temperature for Gumbel noise.
        Returns:
            x: (n, L) long tensor, final sampled sequences.
            xt_traj: (T+1, n, L) long tensor, the full trajectory of x_t from t=1 to t=0.
        Function:
            Generates samples from the model, returning both the final samples and the full trajectory.
        """
        self.model.eval()
        device = self.device
        L = self.model.seq_len

        ts = torch.linspace(1.0, 0.0, steps=T+1, device=device)
        x = torch.full((n, L), self.mask_id, dtype=torch.long, device=device)
        traj = [x.clone()]

        num_transfer_tokens = get_num_transfer_tokens(x==self.mask_id, T)
        for i in range(T):
            t_curr = ts[i]
            t_tensor = torch.full((n,), t_curr, device=device)
            
            # Get model logits
            logits = self.model(x, t_tensor)  # (n, L, V+1)
            
            # Only consider data tokens (exclude mask token)
            logits_data = logits[:, :, :self.mask_id]  # (n, L, V)
            
            # Apply Gumbel noise with temperature
            gumbel_logits = add_gumbel_noise(logits_data, temperature)
            
            # Sample candidate tokens
            candidate_tokens = gumbel_logits.argmax(dim=-1)  # (n, L)
            
            # For each sequence, determine which masked positions to unmask
            for b in range(n):
                # Find masked positions for this sequence
                masked_positions = (x[b] == self.mask_id).nonzero(as_tuple=True)[0]
                
                if len(masked_positions) == 0:
                    continue
                
                # Get number of tokens to transfer at this step
                num_to_transfer = num_transfer_tokens[b, i].item()
                num_to_transfer = min(num_to_transfer, len(masked_positions))
                
                if num_to_transfer == 0:
                    continue
                
                # Get confidence scores for masked positions
                # Use the gumbel_logits values at the sampled tokens as confidence
                confidences = gumbel_logits[b, masked_positions, candidate_tokens[b, masked_positions]]
                
                # Select top-k positions by confidence
                _, top_indices = confidences.topk(num_to_transfer)
                positions_to_unmask = masked_positions[top_indices]
                
                # Replace mask tokens with sampled tokens
                x[b, positions_to_unmask] = candidate_tokens[b, positions_to_unmask]
            
            traj.append(x.clone())
        xt_traj = torch.stack(traj, dim=0)
        return x, xt_traj

# ---------------- Utils ----------------

def sinusoidal_t_embed(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=device))
    ang = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(t[:, :1])], dim=-1)
    return emb

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def eval_kl_by_matrix(data, P):
    Vshape = P.shape
    C = np.zeros(Vshape, dtype=int)
    np.add.at(C, tuple(data[:, i] for i in range(len(Vshape))), 1)
    Q = C / C.sum()
    eps = 1e-8
    Q = np.clip(Q, eps, 1.0)
    P = np.clip(P, eps, 1.0)
    kl = np.sum(P * np.log(P / Q))
    return kl, C

def seq_evaluation(seqs, dataset):
    kl, C = eval_kl_by_matrix(seqs.detach().cpu().numpy(), dataset.P)
    s = seqs.unsqueeze(1)    # [N,1,L]
    p = torch.tensor(dataset.data, device=seqs.device).unsqueeze(0) # [1,10,L]
    eq = (s == p).all(dim=-1)
    acc = eq.any(dim=1).float().mean().item()
    print(f"[{dataset.name}] KL = {kl:.3f}")
    print(f"[{dataset.name}] acc = {acc:.3f}")
    return kl, acc

# ---------------- Checkpoint I/O ----------------

def save_ckpt(path, model, optimizer, steps, best_loss, seed=SEED):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": int(steps),
        "best_loss": float(best_loss),
        "seed": int(seed),
    }
    torch.save(ckpt, path)

def load_ckpt(path, model, optimizer=None, map_location=DEVICE):
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e1:
        print(f"[warn] safe load failed: {e1}\n        retrying with weights_only=False (trusted file only)")
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return {
        "steps": ckpt.get("steps", 0),
        "best_loss": ckpt.get("best_loss", float("inf")),
        "seed": ckpt.get("seed", None),
    }

# ---------------- Main ----------------

def run_one_dataset(dset_name):
    # ---- data ----
    dataset = ToyDatasets(dset_name)
    vocab = dataset.vocab
    seq_len = dataset.seq_len
    outdir = ensure_dir(os.path.join(ROOT_OUTDIR, dset_name))

    # ---- model & diffusion ----
    model = TinyTransformer(vocab+1, seq_len).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    mdm  = MDM(model=model, vocab=vocab, device=DEVICE)

    # ---- resume ----
    last_ckpt = os.path.join(outdir, "last.pt")
    best_ckpt = os.path.join(outdir, "best.pt")
    start_steps, best_loss = 0, float("inf")
    if RESUME and os.path.isfile(last_ckpt):
        extras = load_ckpt(last_ckpt, model, opt, map_location=DEVICE)
        start_steps = int(extras.get("steps", 0))
        best_loss   = float(extras.get("best_loss", float("inf")))
        print(f"[{dset_name}] resume steps={start_steps}, best_loss={best_loss:.6f}")

    # ---- train ----
    total_loss = 0
    rng = trange(start_steps, TOTAL_STEPS, dynamic_ncols=True, desc="Train MDM")
    for step in rng:
        x0 = dataset.make_batch(BATCH_SIZE, DEVICE)
        loss = mdm.train_one_step(x0, opt)
        total_loss += loss.item()
        rng.set_postfix_str(f"loss={loss.item():.4f}")
        if (step + 1) % SAVE_EVERY == 0:
            avg_loss = total_loss / step
            total_loss = 0
            save_ckpt(last_ckpt, model, opt, step+1, min(best_loss, avg_loss), seed=SEED)
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_ckpt(best_ckpt, model, opt, step+1, best_loss, seed=SEED)

    if os.path.isfile(best_ckpt):
        load_ckpt(best_ckpt, model, None, map_location=DEVICE)
        print(f"[{dset_name}] loaded best checkpoint.")

    # ---- sample (final distribution + trajectories) ----
    with torch.no_grad():
        mdm_final, mdm_traj = mdm.sample(n=N_SAMPLES, T=4)
        for xt in mdm_traj:
            print(xt[:5])
        kl, acc = seq_evaluation(mdm_final, dataset)


def main():
    seed_everything(SEED)
    ensure_dir(ROOT_OUTDIR)

    run_one_dataset(dset_name='cont_pairs4')

    print("[done] all datasets processed. See:", ROOT_OUTDIR)

if __name__ == "__main__":
    main()
