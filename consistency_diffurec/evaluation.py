"""
Evaluation utilities.

- evaluate_teacher_truncated: takes the trained teacher and runs a DDIM-style
  reverse loop with `num_steps` steps instead of T. This is the "naive
  truncation" baseline against which the distilled student is compared.

- measure_latency_grid: latency in ms/sample across NFE values for both the
  teacher (truncated DDIM) and the student.
"""
import time
import numpy as np
import torch

from trainer import hrs_and_ndcgs_k
from diffurec import _extract_into_tensor


@torch.no_grad()
def _ddim_step(teacher_diffu, item_rep, x_t_high, t_high, t_low, mask_seq):
    """One deterministic DDIM step from t_high down to t_low using the teacher."""
    x_0_t, _ = teacher_diffu.xstart_model(
        item_rep, x_t_high, teacher_diffu._scale_timesteps(t_high), mask_seq
    )
    sa_h  = _extract_into_tensor(teacher_diffu.sqrt_alphas_cumprod, t_high, x_t_high.shape)
    som_h = _extract_into_tensor(teacher_diffu.sqrt_one_minus_alphas_cumprod, t_high, x_t_high.shape)
    sa_l  = _extract_into_tensor(teacher_diffu.sqrt_alphas_cumprod, t_low, x_t_high.shape)
    som_l = _extract_into_tensor(teacher_diffu.sqrt_one_minus_alphas_cumprod, t_low, x_t_high.shape)
    eps   = (x_t_high - sa_h * x_0_t) / som_h
    return sa_l * x_0_t + som_l * eps, x_0_t


@torch.no_grad()
def teacher_truncated_predict(teacher, sequence, num_steps):
    """
    Run the teacher with `num_steps` DDIM steps and return per-item scores.

    num_steps=1: one forward pass at t = T-1 -> use the predicted x_0 directly.
    num_steps>1: walk the DDIM trajectory T-1 -> ... -> 0 with num_steps points.
    """
    device = sequence.device
    diffu  = teacher.diffu

    # Encode the history exactly as Att_Diffuse_model.forward does.
    item_emb = teacher.item_embeddings(sequence)
    item_emb = teacher.embed_dropout(item_emb)
    item_emb = teacher.LayerNorm(item_emb)
    mask_seq = (sequence > 0).float()

    bs = sequence.size(0)
    H  = item_emb.size(-1)
    T  = diffu.num_timesteps

    x_t = torch.randn(bs, H, device=device)

    if num_steps == 1:
        t = torch.full((bs,), T - 1, device=device, dtype=torch.long)
        x_0, _ = diffu.xstart_model(item_emb, x_t, diffu._scale_timesteps(t), mask_seq)
    else:
        ts = np.linspace(T - 1, 0, num_steps + 1).round().astype(int)
        x_cur = x_t
        x_0   = None
        for i in range(num_steps):
            t_high = torch.full((bs,), int(ts[i]),     device=device, dtype=torch.long)
            t_low  = torch.full((bs,), int(ts[i + 1]), device=device, dtype=torch.long)
            x_cur, x_0 = _ddim_step(diffu, item_emb, x_cur, t_high, t_low, mask_seq)

    return torch.matmul(x_0, teacher.item_embeddings.weight.t())


@torch.no_grad()
def evaluate_teacher_truncated(teacher, loader, num_steps, device, ks=(5, 10, 20)):
    """HR@k / NDCG@k for the teacher run with truncated DDIM at `num_steps`."""
    teacher.eval()
    acc = {f'HR@{k}':   [] for k in ks}
    acc.update({f'NDCG@{k}': [] for k in ks})
    for batch in loader:
        seq, target = [x.to(device) for x in batch]
        scores = teacher_truncated_predict(teacher, seq, num_steps)
        m = hrs_and_ndcgs_k(scores, target, list(ks))
        for k, v in m.items():
            acc[k].append(v)
    return {k: round(float(np.mean(v)) * 100, 4) for k, v in acc.items()}


@torch.no_grad()
def measure_latency_grid(teacher, student, sample_batch, device,
                         nfe_grid, n_warmup=10, n_runs=50):
    """ms/sample for teacher (truncated) and student at each NFE in the grid."""
    seq, target = [x.to(device) for x in sample_batch]
    bs = seq.size(0)
    out = {'student': {}, 'teacher_truncated': {}, 'teacher_full': None}

    teacher.eval()
    student.eval()

    def _time(fn):
        for _ in range(n_warmup):
            fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_runs):
            fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return (time.time() - t0) * 1000.0 / n_runs / bs

    for nfe in nfe_grid:
        out['student'][str(nfe)] = _time(lambda n=nfe: student.predict_scores(seq, num_steps=n))
        out['teacher_truncated'][str(nfe)] = _time(
            lambda n=nfe: teacher_truncated_predict(teacher, seq, num_steps=n)
        )

    def _teacher_full():
        _, rep, *_ = teacher(seq, target, train_flag=False)
        teacher.diffu_rep_pre(rep)
    out['teacher_full'] = _time(_teacher_full)

    return out