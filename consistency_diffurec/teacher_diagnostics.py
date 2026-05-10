"""
Teacher diagnostics v2 — rigorous statistical version.

What changed vs v1
------------------
* Check 1+3: Replaced loose "non-trivial variance" verdict with a quantitative
  ratio (run-std / embedding-std) and a 4-tier scale: near-deterministic /
  weakly / moderately / highly stochastic.
* Check 2: Added MEAN DRIFT across timesteps. Variance-vs-t alone misses what
  matters for trajectory weighting: whether the *expected* prediction shifts
  with t. We now measure ‖mean(x₀|t_min) − mean(x₀|t_max)‖ / emb-scale.
* Check 6: Replaced ad-hoc thresholds with proper statistics:
    – Wilson 95% CI for per-bin HR@10
    – Chi-squared test on bin × {hit, miss} contingency table
    – Linear regression slope-test of HR vs bin midpoint length
    – Mann-Kendall non-parametric monotonicity test
    – Bootstrap 95% CI on max/min variance ratio (user-level resampling)
  LACD is declared justified ONLY IF four criteria all pass per dataset
  AND the trend direction agrees across datasets (cross-dataset rule).

Usage
-----
    !python teacher_diagnostics_v2.py \
        --dataset toys \
        --data_root /content/diffurec-distillation/datasets/data \
        --teacher_ckpt /content/.../teacher_toys.pt \
        --src_path /content/diffurec-distillation/original_diffurec \
        --out_json diagnostics_v2/toys.json \
        --out_md   diagnostics_v2/toys.md

Run on both datasets, then verify cross-dataset agreement manually
(or use compare_diagnostics_v2() at the bottom of this file).
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from scipy import stats as sst
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ===================================================================== #
#  Statistical helpers                                                  #
# ===================================================================== #
def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def chi2_contingency(observed):
    """Chi-squared test of independence on a 2D table. Returns (chi2, df, p)."""
    obs = np.asarray(observed, dtype=float)
    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    total = obs.sum()
    expected = row_sum @ col_sum / max(total, 1e-12)
    chi2 = ((obs - expected) ** 2 / np.maximum(expected, 1e-12)).sum()
    df = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    if HAVE_SCIPY:
        p = float(1.0 - sst.chi2.cdf(chi2, df))
    else:
        p = float('nan')
    return float(chi2), int(df), p


def linregress(x, y):
    """Simple linear regression returning (slope, intercept, r2, p_slope)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if HAVE_SCIPY:
        r = sst.linregress(x, y)
        return float(r.slope), float(r.intercept), float(r.rvalue ** 2), float(r.pvalue)
    n = len(x)
    if n < 2:
        return 0.0, float(y.mean()) if n else 0.0, 0.0, float('nan')
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    sxy = ((x - xm) * (y - ym)).sum()
    slope = sxy / max(sxx, 1e-12)
    inter = ym - slope * xm
    y_pred = slope * x + inter
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return float(slope), float(inter), float(r2), float('nan')


def mann_kendall(values):
    """Mann-Kendall test for monotonic trend. Returns (S, normalized, p)."""
    n = len(values)
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += int(np.sign(values[j] - values[i]))
    max_s = n * (n - 1) // 2
    norm = s / max_s if max_s > 0 else 0.0
    if HAVE_SCIPY and n >= 4:
        var_s = n * (n - 1) * (2 * n + 5) / 18.0
        z = (s - np.sign(s)) / np.sqrt(var_s) if var_s > 0 else 0.0
        p = float(2 * (1 - sst.norm.cdf(abs(z))))
    else:
        p = float('nan')
    return int(s), float(norm), p


def bootstrap_var_ratio(per_bin_user_vars, n_boot=1000, seed=0):
    """Bootstrap 95% CI on max/min mean-variance ratio across bins.

    per_bin_user_vars: list of arrays — per-user variance values for each bin.
    Resamples users within each bin (with replacement), computes the bin means,
    then the max/min ratio. Returns (lo_2.5, median, hi_97.5).
    """
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        bin_means = []
        for vals in per_bin_user_vars:
            vals = np.asarray(vals)
            sample = rng.choice(vals, size=len(vals), replace=True)
            bin_means.append(sample.mean())
        bm = np.array(bin_means)
        ratios.append(bm.max() / max(bm.min(), 1e-12))
    arr = np.array(ratios)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 50.0)), float(np.percentile(arr, 97.5))


# ===================================================================== #
#  Path setup + loading (same as v1)                                    #
# ===================================================================== #
def _setup_imports(explicit_src=None):
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if explicit_src:
        candidates.append(explicit_src)
    candidates.extend([
        os.path.join(here, '..', 'original_diffurec'),
        os.path.join(here, 'original_diffurec'),
        os.path.join(here, '..', 'src'),
        here,
    ])
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(c) and os.path.exists(os.path.join(c, 'diffurec.py')):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    raise RuntimeError(f'Could not locate diffurec.py in: {candidates}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',      default='toys')
    p.add_argument('--data_root',    required=True)
    p.add_argument('--teacher_ckpt', required=True)
    p.add_argument('--src_path',     default=None)
    p.add_argument('--device',       default='cuda')
    p.add_argument('--max_len',      type=int, default=50)
    p.add_argument('--batch_size',   type=int, default=512)
    p.add_argument('--hidden_size',  type=int, default=128)
    p.add_argument('--dropout',      type=float, default=0.1)
    p.add_argument('--emb_dropout',  type=float, default=0.3)
    p.add_argument('--hidden_act',   default='gelu')
    p.add_argument('--num_blocks',   type=int, default=4)
    p.add_argument('--diffusion_steps',     type=int,   default=32)
    p.add_argument('--lambda_uncertainty',  type=float, default=0.001)
    p.add_argument('--noise_schedule',      default='trunc_lin')
    p.add_argument('--rescale_timesteps',   default=True)
    p.add_argument('--schedule_sampler_name', default='lossaware')
    p.add_argument('--K_runs', type=int, default=12)
    p.add_argument('--n_samples_per_bin', type=int, default=300)
    p.add_argument('--n_boot', type=int, default=1000)
    p.add_argument('--out_json', default='diagnostics_v2/teacher.json')
    p.add_argument('--out_md',   default=None)
    return p.parse_args()


def load_teacher_and_data(args):
    from utils import Data_Test
    from model import create_model_diffu, Att_Diffuse_model
    path = os.path.join(args.data_root, args.dataset, 'dataset.pkl')
    with open(path, 'rb') as f:
        data_raw = pickle.load(f)
    args.item_num = len(data_raw['smap'])
    device = torch.device(args.device if torch.cuda.is_available() or args.device != 'cuda' else 'cpu')
    teacher = Att_Diffuse_model(create_model_diffu(args), args).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    test_loader = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'],
                            args).get_pytorch_dataloaders()
    return teacher, test_loader, data_raw, device


def encode_history(teacher, sequence):
    item_emb = teacher.item_embeddings(sequence)
    item_emb = teacher.embed_dropout(item_emb)
    item_emb = teacher.LayerNorm(item_emb)
    mask_seq = (sequence > 0).float()
    return item_emb, mask_seq


# ===================================================================== #
#  CHECK 0 — sanity (HR/NDCG vs paper)                                  #
# ===================================================================== #
@torch.no_grad()
def check_0_sanity(teacher, test_loader, device, dataset_name):
    from trainer import hrs_and_ndcgs_k
    print('\n' + '=' * 72)
    print('CHECK 0 — HR/NDCG on test (sanity vs DiffuRec paper)')
    print('=' * 72)
    ks = [5, 10, 20]
    acc = {f'HR@{k}': [] for k in ks}
    acc.update({f'NDCG@{k}': [] for k in ks})
    for batch in test_loader:
        seq, target = [x.to(device) for x in batch]
        _, rep_diffu, *_ = teacher(seq, target, train_flag=False)
        scores = teacher.diffu_rep_pre(rep_diffu)
        m = hrs_and_ndcgs_k(scores, target, ks)
        for k, v in m.items():
            acc[k].append(v)
    metrics = {k: round(float(np.mean(v)) * 100, 4) for k, v in acc.items()}
    print(f'  {metrics}')
    PAPER = {
        'toys':           {'HR@10': 7.4587,  'NDCG@10': 4.7724},
        'amazon_beauty':  {'HR@10': 7.9068,  'NDCG@10': 4.7494},
        'ml-1m':          {'HR@10': 26.2647, 'NDCG@10': 14.7909},
        'steam':          {'HR@10': 10.7520, 'NDCG@10': 5.5981},
    }
    if dataset_name in PAPER:
        ref = PAPER[dataset_name]
        gap_hr = (metrics['HR@10']  - ref['HR@10'])  / ref['HR@10']  * 100
        print(f'  Paper:  HR@10={ref["HR@10"]:.4f}    Your gap: {gap_hr:+.2f}%')
        verdict = '✓ within 10% of paper' if abs(gap_hr) < 10 else '⚠️  outside 10% band'
        print(f'  VERDICT: {verdict}')
        metrics['_paper_gap_HR10_pct'] = gap_hr
    return metrics


# ===================================================================== #
#  CHECK 1+3 — quantitative variance characterization                   #
# ===================================================================== #
@torch.no_grad()
def check_1_3_quantitative(teacher, test_loader, device, K=12):
    print('\n' + '=' * 72)
    print(f'CHECK 1+3 — Variance characterization (K={K} runs on a long history)')
    print('=' * 72)

    seq_batch, target_batch = next(iter(test_loader))
    seq_batch = seq_batch.to(device); target_batch = target_batch.to(device)
    lens = (seq_batch > 0).sum(dim=1).tolist()
    idx = max(range(len(lens)), key=lambda i: lens[i])
    sample_seq = seq_batch[idx:idx + 1]
    sample_tgt = target_batch[idx:idx + 1]

    x0_runs, top1_runs = [], []
    for _ in range(K):
        _, rep, *_ = teacher(sample_seq, sample_tgt, train_flag=False)
        x0_runs.append(rep[0].cpu().numpy())
        scores = teacher.diffu_rep_pre(rep)
        top1_runs.append(int(scores.argmax(dim=-1)[0].item()))
    x0 = np.stack(x0_runs)
    run_std = float(x0.std(axis=0, ddof=1).mean())

    emb_w = teacher.item_embeddings.weight.detach().cpu().numpy()
    emb_std = float(emb_w.std(axis=0).mean())
    ratio = run_std / max(emb_std, 1e-12)

    # Quantitative tier
    if ratio < 1e-3:
        tier = 'NEAR-DETERMINISTIC'
        reading = ('predictions vary by less than 0.1% of the embedding scale; '
                   'the diffusion noise has essentially no observable effect on output.')
    elif ratio < 1e-2:
        tier = 'WEAKLY STOCHASTIC'
        reading = 'measurable but small variability (≤1% of embedding scale).'
    elif ratio < 1e-1:
        tier = 'MODERATELY STOCHASTIC'
        reading = 'noticeable run-to-run variation, on the order of 1–10% of embedding scale.'
    else:
        tier = 'HIGHLY STOCHASTIC'
        reading = 'predictions span a substantial fraction of the embedding scale.'

    print(f'  Sample length: {lens[idx]}')
    print(f'  Run-to-run std of x_0:        {run_std:.6e}')
    print(f'  Item embedding std (per-dim): {emb_std:.6e}')
    print(f'  Ratio (run / embedding):      {ratio:.3e}')
    print(f'  Unique top-1 across K runs:   {len(set(top1_runs))} / {K}')
    print(f'\n  TIER: {tier}')
    print(f'  Reading: {reading}')
    if tier == 'NEAR-DETERMINISTIC':
        print('\n  IMPLICATION FOR DISTILLATION: The teacher is approximately a function')
        print('  f(history) → x_0, with x_t and t having near-zero influence. Consistency')
        print('  distillation is therefore expected to succeed at NFE=1, but for a less')
        print('  exciting reason than the original Consistency Models paper assumes.')

    return {
        'sample_length': int(lens[idx]),
        'run_std': run_std,
        'emb_std': emb_std,
        'ratio': ratio,
        'tier':  tier,
        'unique_top1': len(set(top1_runs)),
        'top1_items':  top1_runs,
    }


# ===================================================================== #
#  CHECK 2 v2 — variance AND mean drift across t                        #
# ===================================================================== #
@torch.no_grad()
def check_2_drift(teacher, test_loader, device, K=12, T=32):
    print('\n' + '=' * 72)
    print(f'CHECK 2 v2 — Variance and MEAN DRIFT across timestep t (K={K})')
    print('=' * 72)

    seq_batch, _ = next(iter(test_loader))
    seq_batch = seq_batch.to(device)
    lens = (seq_batch > 0).sum(dim=1).tolist()
    idx = max(range(len(lens)), key=lambda i: lens[i])
    sample_seq = seq_batch[idx:idx + 1]
    item_emb, mask_seq = encode_history(teacher, sample_seq)
    diffu = teacher.diffu

    t_grid = sorted({1, max(1, T // 8), T // 4, T // 2, 3 * T // 4, T - 1})
    rows = []
    means_per_t = {}
    for t_val in t_grid:
        t = torch.full((1,), int(t_val), device=device, dtype=torch.long)
        runs = []
        for _ in range(K):
            x_t = torch.randn_like(item_emb[:, -1, :])
            x_0, _ = diffu.xstart_model(item_emb, x_t,
                                        diffu._scale_timesteps(t), mask_seq)
            runs.append(x_0[0].cpu().numpy())
        runs = np.stack(runs)
        m = runs.mean(axis=0)
        means_per_t[t_val] = m
        rows.append({'t': int(t_val),
                     'mean_std': float(runs.std(axis=0, ddof=1).mean()),
                     'mean_norm': float(np.linalg.norm(m))})
        print(f"  t={t_val:>3}   mean_std={rows[-1]['mean_std']:.6e}   ‖mean(x₀)‖={rows[-1]['mean_norm']:.4f}")

    # Mean drift across t
    t_min, t_max = min(t_grid), max(t_grid)
    drift_t_minmax = float(np.linalg.norm(means_per_t[t_min] - means_per_t[t_max]))

    # Pairwise drift matrix (max distance between any two t's)
    ts = sorted(means_per_t.keys())
    max_pair_drift = 0.0
    argmax_pair = (None, None)
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            d = float(np.linalg.norm(means_per_t[ts[i]] - means_per_t[ts[j]]))
            if d > max_pair_drift:
                max_pair_drift = d
                argmax_pair = (ts[i], ts[j])

    emb_w = teacher.item_embeddings.weight.detach().cpu().numpy()
    emb_norm = float(np.linalg.norm(emb_w, axis=1).mean())
    drift_ratio = max_pair_drift / max(emb_norm, 1e-12)

    print(f'\n  Mean prediction drift across t:')
    print(f'    L2 between mean(t={t_min}) and mean(t={t_max}): {drift_t_minmax:.6e}')
    print(f'    Max L2 across all (t_i, t_j) pairs:           {max_pair_drift:.6e}  (pair {argmax_pair})')
    print(f'    Mean ‖item embedding‖:                        {emb_norm:.4f}')
    print(f'    Drift ratio (max_pair_drift / emb_norm):      {drift_ratio:.3e}')

    if drift_ratio < 1e-3:
        verdict = '⚠️  Mean prediction is essentially t-INDEPENDENT. Trajectory weighting in the consistency loss is *not* empirically motivated by this teacher.'
    elif drift_ratio < 1e-2:
        verdict = '~  Drift is small but measurable (~0.1–1% of embedding scale). Trajectory weighting has weak justification.'
    else:
        verdict = '✓  Mean prediction shifts substantially with t. Trajectory weighting is motivated.'
    print(f'\n  VERDICT: {verdict}')

    return {
        'per_t': rows,
        'drift_t_minmax': drift_t_minmax,
        'max_pair_drift': max_pair_drift,
        'argmax_pair':    list(argmax_pair),
        'embedding_norm': emb_norm,
        'drift_ratio':    drift_ratio,
        'verdict':        verdict,
    }


# ===================================================================== #
#  CHECK 4 — full reverse loop vs single xstart_model (unchanged)       #
# ===================================================================== #
@torch.no_grad()
def check_4_train_flag(teacher, test_loader, device, K=8):
    print('\n' + '=' * 72)
    print('CHECK 4 — Full reverse loop vs single xstart_model call')
    print('=' * 72)
    seq_batch, target_batch = next(iter(test_loader))
    seq_batch    = seq_batch.to(device)
    target_batch = target_batch.to(device)
    sample_seq = seq_batch[:1]; sample_tgt = target_batch[:1]
    item_emb, mask_seq = encode_history(teacher, sample_seq)
    T = teacher.diffu.num_timesteps

    full_runs, single_runs = [], []
    for _ in range(K):
        _, rep_diffu, *_ = teacher(sample_seq, sample_tgt, train_flag=False)
        full_runs.append(rep_diffu[0].cpu().numpy())
        x_t = torch.randn_like(item_emb[:, -1, :])
        t = torch.full((1,), T - 1, device=device, dtype=torch.long)
        x_0, _ = teacher.diffu.xstart_model(item_emb, x_t,
                                            teacher.diffu._scale_timesteps(t), mask_seq)
        single_runs.append(x_0[0].cpu().numpy())
    full_arr   = np.stack(full_runs)
    single_arr = np.stack(single_runs)
    full_std   = float(full_arr.std(axis=0, ddof=1).mean())
    single_std = float(single_arr.std(axis=0, ddof=1).mean())
    centroid_dist = float(np.linalg.norm(full_arr.mean(0) - single_arr.mean(0)))
    print(f'  Full reverse (T={T}):  std={full_std:.6e}')
    print(f'  Single xstart at T-1:  std={single_std:.6e}')
    print(f'  L2 between mean predictions: {centroid_dist:.6e}')
    return {'full_std': full_std, 'single_std': single_std, 'centroid_dist': centroid_dist}


# ===================================================================== #
#  CHECK 5 — λ_uncertainty sweep (full test set HR for trained λ)       #
# ===================================================================== #
@torch.no_grad()
def check_5_lambda(teacher, test_loader, device, K=8):
    from trainer import hrs_and_ndcgs_k
    print('\n' + '=' * 72)
    print('CHECK 5 — λ_uncertainty sweep')
    print('=' * 72)

    seq_batch, target_batch = next(iter(test_loader))
    seq_batch    = seq_batch.to(device)
    target_batch = target_batch.to(device)
    sample = seq_batch[:1]; sample_tgt = target_batch[:1]

    saved = teacher.diffu.xstart_model.lambda_uncertainty
    sweep = [0.0, 0.001, 0.01, 0.1, 1.0]
    rows = []
    print(f"\n  {'lambda':>8}  {'std_x0':>12}  {'HR@10 (1 batch)':>16}")
    for val in sweep:
        teacher.diffu.xstart_model.lambda_uncertainty = float(val)
        runs = []
        for _ in range(K):
            _, rep, *_ = teacher(sample, sample_tgt, train_flag=False)
            runs.append(rep[0].cpu().numpy())
        runs = np.stack(runs)
        v_std = float(runs.std(axis=0, ddof=1).mean())
        _, rep, *_ = teacher(seq_batch, target_batch, train_flag=False)
        scores = teacher.diffu_rep_pre(rep)
        m = hrs_and_ndcgs_k(scores, target_batch, [10])
        hr10 = round(float(m['HR@10']) * 100, 4)
        rows.append({'lambda': val, 'std_x0': v_std, 'HR@10': hr10})
        print(f"  {val:>8g}  {v_std:>12.6e}  {hr10:>16.4f}")
    teacher.diffu.xstart_model.lambda_uncertainty = saved
    return {'sweep': rows, 'trained_lambda': saved}


# ===================================================================== #
#  CHECK 6 v2 — LACD diagnostic with proper statistics                  #
# ===================================================================== #
@torch.no_grad()
def check_6_lacd_v2(teacher, test_loader, device, K=8, n_per_bin=300, n_boot=1000):
    from trainer import hrs_and_ndcgs_k
    print('\n' + '=' * 72)
    print(f'CHECK 6 v2 — LACD diagnostic with rigorous statistics')
    print(f'              (n={n_per_bin} per bin, K={K} runs/sample, {n_boot} bootstrap)')
    print('=' * 72)

    # Collect all (seq, target, length)
    all_seqs, all_targets, all_lens = [], [], []
    for seq, tgt in test_loader:
        for i in range(seq.size(0)):
            all_seqs.append(seq[i])
            all_targets.append(tgt[i])
            all_lens.append(int((seq[i] > 0).sum().item()))
    all_lens = np.array(all_lens)
    print(f'  Total test users: {len(all_lens)}')
    print(f'  Length stats: min={all_lens.min()}  median={int(np.median(all_lens))}  '
          f'max={all_lens.max()}  mean={all_lens.mean():.1f}')

    q = np.quantile(all_lens, [0.25, 0.5, 0.75])
    bins_def = [
        ('Q1 (shortest)', np.where(all_lens <= q[0])[0]),
        ('Q2',            np.where((all_lens > q[0]) & (all_lens <= q[1]))[0]),
        ('Q3',            np.where((all_lens > q[1]) & (all_lens <= q[2]))[0]),
        ('Q4 (longest)',  np.where(all_lens > q[2])[0]),
    ]

    rng = np.random.default_rng(0)
    rows = []
    per_bin_user_vars = []
    bin_midpoints = []

    print()
    print('  Per-bin statistics:')
    print(f'  {"Bin":<14}  {"len range":<12}  {"n":<5}  {"hits":<5}  '
          f'{"HR@10":<8}  {"95% Wilson CI":<18}  {"var_x0":<14}')
    print('  ' + '-' * 90)

    for name, idxs in bins_def:
        if len(idxs) == 0:
            continue
        sel = rng.choice(idxs, size=min(n_per_bin, len(idxs)), replace=False)
        seqs = torch.stack([all_seqs[i] for i in sel]).to(device)
        tgts = torch.stack([all_targets[i] for i in sel]).to(device)

        # Per-user variance via K runs
        runs = []
        for _ in range(K):
            _, rep, *_ = teacher(seqs, tgts, train_flag=False)
            runs.append(rep.cpu().numpy())
        runs = np.stack(runs)                                   # (K, n_sel, H)
        per_user_var = runs.std(axis=0, ddof=1).mean(axis=-1)   # (n_sel,)
        mean_var = float(per_user_var.mean())

        # Per-user hit@10 (single deterministic-ish run is fine; teacher near-determ)
        _, rep, *_ = teacher(seqs, tgts, train_flag=False)
        scores = teacher.diffu_rep_pre(rep)
        _, top10 = torch.topk(scores, k=10, dim=-1)
        hits_per_user = (top10 == tgts).any(dim=-1).cpu().numpy().astype(int)
        n_sel = int(len(sel))
        hits = int(hits_per_user.sum())
        hr10 = hits / n_sel
        ci_lo, ci_hi = wilson_ci(hits, n_sel)

        len_min = int(all_lens[sel].min())
        len_max = int(all_lens[sel].max())
        midpoint = float(all_lens[sel].mean())
        rows.append({
            'bin': name, 'len_min': len_min, 'len_max': len_max,
            'len_mean': midpoint,
            'n': n_sel, 'hits': hits, 'HR@10': hr10 * 100,
            'wilson_lo_pct': ci_lo * 100, 'wilson_hi_pct': ci_hi * 100,
            'mean_var': mean_var,
        })
        per_bin_user_vars.append(per_user_var)
        bin_midpoints.append(midpoint)
        print(f'  {name:<14}  {f"{len_min}-{len_max}":<12}  {n_sel:<5}  {hits:<5}  '
              f'{hr10*100:<8.2f}  [{ci_lo*100:>5.2f}, {ci_hi*100:>5.2f}]    '
              f'{mean_var:<14.6e}')

    # ---- Cross-bin tests ----
    hits_arr = np.array([r['hits'] for r in rows])
    n_arr = np.array([r['n'] for r in rows])
    misses_arr = n_arr - hits_arr
    contingency = np.column_stack([hits_arr, misses_arr])

    chi2, df, p_chi2 = chi2_contingency(contingency)
    hr_arr = np.array([r['HR@10'] for r in rows])
    bin_idx = np.arange(1, len(rows) + 1)
    slope_idx, _, r2_idx, p_idx = linregress(bin_idx, hr_arr)
    slope_len, _, r2_len, p_len = linregress(np.array(bin_midpoints), hr_arr)
    s_mk, norm_mk, p_mk = mann_kendall(hr_arr.tolist())
    var_lo, var_med, var_hi = bootstrap_var_ratio(per_bin_user_vars,
                                                  n_boot=n_boot, seed=0)
    bin_means_var = [v.mean() for v in per_bin_user_vars]
    var_ratio_point = max(bin_means_var) / max(min(bin_means_var), 1e-12)

    print()
    print('  Cross-bin tests:')
    print(f'    Chi-squared (HR homogeneity across bins):')
    p_chi_str = f'{p_chi2:.4g}' if not np.isnan(p_chi2) else 'n/a (no scipy)'
    print(f'      χ² = {chi2:.3f},  df = {df},  p = {p_chi_str}')
    print(f'    Linear regression HR ~ bin_index:')
    print(f'      slope = {slope_idx:+.3f},  R² = {r2_idx:.3f},  p = '
          f'{(f"{p_idx:.4g}" if not np.isnan(p_idx) else "n/a")}')
    print(f'    Linear regression HR ~ mean_length:')
    print(f'      slope = {slope_len:+.4f},  R² = {r2_len:.3f},  p = '
          f'{(f"{p_len:.4g}" if not np.isnan(p_len) else "n/a")}')
    print(f'    Mann-Kendall (monotonicity of HR vs bin index):')
    print(f'      S = {s_mk},  S/S_max = {norm_mk:+.3f},  p = '
          f'{(f"{p_mk:.4g}" if not np.isnan(p_mk) else "n/a")}')
    print(f'    Bootstrap 95% CI on max/min variance ratio:')
    print(f'      point = {var_ratio_point:.3f},  95% CI = [{var_lo:.3f}, {var_hi:.3f}]')

    # ---- Strict LACD criteria ----
    crit = {}
    crit['1_chi2_significant']        = (not np.isnan(p_chi2) and p_chi2 < 0.05)
    crit['2_monotonic']               = (abs(norm_mk) > 0.5)
    crit['3_variance_effect_size']    = (var_lo > 1.2)
    crit['4_HR_gap_practical']        = (hr_arr.max() - hr_arr.min() > 5.0)

    print()
    print('  STRICT LACD CRITERIA (all four must pass):')
    print(f'    [{"✓" if crit["1_chi2_significant"]    else "✗"}] (1) Chi-squared p < 0.05'
          f'                          ({p_chi_str})')
    print(f'    [{"✓" if crit["2_monotonic"]           else "✗"}] (2) Mann-Kendall |S/S_max| > 0.5'
          f'                  ({norm_mk:+.3f})')
    print(f'    [{"✓" if crit["3_variance_effect_size"] else "✗"}] (3) Variance ratio bootstrap CI lower > 1.2'
          f'   ([{var_lo:.3f}, {var_hi:.3f}])')
    print(f'    [{"✓" if crit["4_HR_gap_practical"]    else "✗"}] (4) HR@10 gap > 5 percentage points'
          f'             ({hr_arr.max()-hr_arr.min():.2f} pp)')

    n_pass = sum(crit.values())
    print()
    if n_pass == 4:
        decision = 'JUSTIFIED on this dataset (verify cross-dataset agreement before adopting LACD).'
    else:
        failing = [k for k, v in crit.items() if not v]
        decision = f'NOT JUSTIFIED — failed criteria: {failing}.'
    print(f'  PER-DATASET DECISION: {decision}')

    if not np.isnan(p_chi2) and p_chi2 < 0.05 and abs(norm_mk) <= 0.5:
        print('  NOTE: Bins differ statistically but the difference is NOT monotonic '
              'with length; this is a non-length-aware artifact, not a LACD signal.')

    return {
        'quartile_thresholds': q.tolist(),
        'bins':                rows,
        'tests': {
            'chi2': {'stat': chi2, 'df': df, 'p': p_chi2},
            'linreg_bin_idx': {'slope': slope_idx, 'r2': r2_idx, 'p': p_idx},
            'linreg_length':  {'slope': slope_len, 'r2': r2_len, 'p': p_len},
            'mann_kendall':   {'S': s_mk, 'S_norm': norm_mk, 'p': p_mk},
            'var_ratio_bootstrap': {'point': var_ratio_point,
                                    'lo_2.5': var_lo, 'med_50': var_med, 'hi_97.5': var_hi},
            'hr_gap_pp': float(hr_arr.max() - hr_arr.min()),
        },
        'criteria':  {k: bool(v) for k, v in crit.items()},
        'n_passed':  int(n_pass),
        'decision':  decision,
        'slope_sign': int(np.sign(slope_len)),
    }


# ===================================================================== #
#  Markdown export                                                      #
# ===================================================================== #
def write_markdown(results, path):
    Path(os.path.dirname(path) or '.').mkdir(parents=True, exist_ok=True)
    lines = []
    ds = results['dataset']
    lines.append(f'# Teacher diagnostics — {ds}\n')

    s = results['check_0_sanity']
    lines.append('## Check 0 — sanity vs DiffuRec paper\n')
    lines.append(f'- HR@10 = **{s.get("HR@10")}**, gap vs paper = '
                 f'**{s.get("_paper_gap_HR10_pct", "n/a")}%**\n')

    s = results['check_1_3_quantitative']
    lines.append('## Check 1+3 — variance characterization\n')
    lines.append(f'| Quantity | Value |\n|---|---|\n'
                 f'| run-to-run std of x₀ | {s["run_std"]:.3e} |\n'
                 f'| item embedding std (per-dim) | {s["emb_std"]:.3e} |\n'
                 f'| ratio (run / emb) | **{s["ratio"]:.3e}** |\n'
                 f'| tier | **{s["tier"]}** |\n'
                 f'| unique top-1 across {len(s["top1_items"])} runs | {s["unique_top1"]} |\n')

    s = results['check_2_drift']
    lines.append('## Check 2 — variance + mean drift across t\n')
    lines.append(f'- Max pairwise mean drift: **{s["max_pair_drift"]:.3e}**\n')
    lines.append(f'- Drift / embedding norm: **{s["drift_ratio"]:.3e}**\n')
    lines.append(f'- Verdict: {s["verdict"]}\n')

    s = results['check_6_lacd_v2']
    lines.append('## Check 6 — LACD diagnostic\n\n### Per-bin\n')
    lines.append('| Bin | len range | n | hits | HR@10 (%) | 95% Wilson CI | mean var x₀ |\n')
    lines.append('|---|---|---|---|---|---|---|\n')
    for r in s['bins']:
        lines.append(f'| {r["bin"]} | {r["len_min"]}-{r["len_max"]} | {r["n"]} | '
                     f'{r["hits"]} | {r["HR@10"]:.2f} | '
                     f'[{r["wilson_lo_pct"]:.2f}, {r["wilson_hi_pct"]:.2f}] | '
                     f'{r["mean_var"]:.3e} |\n')

    t = s['tests']
    lines.append('\n### Cross-bin tests\n')
    lines.append(f'- χ² = {t["chi2"]["stat"]:.3f}, df = {t["chi2"]["df"]}, p = {t["chi2"]["p"]}\n')
    lines.append(f'- Linear slope HR vs length: {t["linreg_length"]["slope"]:+.4f} (R²={t["linreg_length"]["r2"]:.3f}, p={t["linreg_length"]["p"]})\n')
    lines.append(f'- Mann-Kendall S/S_max = {t["mann_kendall"]["S_norm"]:+.3f}, p = {t["mann_kendall"]["p"]}\n')
    lines.append(f'- Variance ratio bootstrap 95% CI: [{t["var_ratio_bootstrap"]["lo_2.5"]:.3f}, {t["var_ratio_bootstrap"]["hi_97.5"]:.3f}]\n')
    lines.append(f'- HR@10 gap: {t["hr_gap_pp"]:.2f} pp\n')

    lines.append('\n### Criteria\n')
    for k, v in s['criteria'].items():
        lines.append(f'- {"✓" if v else "✗"} {k}\n')
    lines.append(f'\n**Decision:** {s["decision"]}\n')

    with open(path, 'w') as f:
        f.write(''.join(lines))
    print(f'\n[md] wrote {path}')


# ===================================================================== #
#  Cross-dataset comparator (call after running on multiple datasets)   #
# ===================================================================== #
def compare_diagnostics_v2(json_paths):
    """Take JSONs from this script run on multiple datasets; report whether
    LACD is justified across all of them with consistent slope direction."""
    print('\n' + '=' * 72)
    print('CROSS-DATASET LACD VERIFICATION')
    print('=' * 72)
    per_ds = []
    for p in json_paths:
        with open(p) as f:
            r = json.load(f)
        c6 = r['check_6_lacd_v2']
        per_ds.append({
            'dataset':    r['dataset'],
            'n_passed':   c6['n_passed'],
            'decision':   c6['decision'],
            'slope_sign': c6.get('slope_sign', 0),
        })
        print(f'  {r["dataset"]}: {c6["n_passed"]}/4 criteria passed, slope sign = {c6.get("slope_sign")}')

    all_pass = all(d['n_passed'] == 4 for d in per_ds)
    same_sign = len({d['slope_sign'] for d in per_ds}) == 1
    print()
    if all_pass and same_sign:
        print('  CROSS-DATASET DECISION: ✓ LACD is justified.')
    elif all_pass and not same_sign:
        print('  CROSS-DATASET DECISION: ✗ Per-dataset criteria pass but slope direction')
        print('     disagrees across datasets — no consistent length-aware signal.')
    else:
        print('  CROSS-DATASET DECISION: ✗ LACD is NOT justified.')
        print('     Recommendation: Variant 1 (Position + Trajectory weighting).')


# ===================================================================== #
#  Main                                                                 #
# ===================================================================== #
def main():
    args = parse_args()
    src = _setup_imports(args.src_path)
    print(f'[setup] DiffuRec source: {src}')
    print(f'[setup] scipy available: {HAVE_SCIPY}')

    Path(os.path.dirname(args.out_json) or '.').mkdir(parents=True, exist_ok=True)
    teacher, test_loader, _, device = load_teacher_and_data(args)
    print(f'[setup] teacher loaded; item_num={args.item_num}, device={device}')

    results = {'dataset': args.dataset, 'teacher_ckpt': args.teacher_ckpt}
    t0 = time.time()
    results['check_0_sanity']           = check_0_sanity(teacher, test_loader, device, args.dataset)
    results['check_1_3_quantitative']   = check_1_3_quantitative(teacher, test_loader, device, K=args.K_runs)
    results['check_2_drift']            = check_2_drift(teacher, test_loader, device,
                                                        K=args.K_runs, T=args.diffusion_steps)
    results['check_4_train_flag']       = check_4_train_flag(teacher, test_loader, device,
                                                             K=max(4, args.K_runs // 2))
    results['check_5_lambda']           = check_5_lambda(teacher, test_loader, device,
                                                         K=max(4, args.K_runs // 2))
    results['check_6_lacd_v2']          = check_6_lacd_v2(teacher, test_loader, device,
                                                          K=max(4, args.K_runs // 2),
                                                          n_per_bin=args.n_samples_per_bin,
                                                          n_boot=args.n_boot)
    results['_elapsed_sec'] = time.time() - t0

    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\n[Done] saved JSON -> {args.out_json}  (elapsed {results["_elapsed_sec"]:.1f}s)')

    if args.out_md:
        write_markdown(results, args.out_md)

    print('\n' + '=' * 72)
    print('SUMMARY')
    print('=' * 72)
    print(f'  Check 0 (sanity):    HR@10 gap vs paper = '
          f'{results["check_0_sanity"].get("_paper_gap_HR10_pct", "n/a")}%')
    print(f'  Check 1+3 (var):     tier = {results["check_1_3_quantitative"]["tier"]}, '
          f'ratio = {results["check_1_3_quantitative"]["ratio"]:.2e}')
    print(f'  Check 2 (drift):     drift_ratio = {results["check_2_drift"]["drift_ratio"]:.2e}')
    print(f'                       {results["check_2_drift"]["verdict"]}')
    print(f'  Check 6 (LACD):      {results["check_6_lacd_v2"]["n_passed"]}/4 criteria, '
          f'{results["check_6_lacd_v2"]["decision"]}')
    print('\n  After running this on a SECOND dataset, run:')
    print('    from teacher_diagnostics_v2 import compare_diagnostics_v2')
    print('    compare_diagnostics_v2([\'path1.json\', \'path2.json\'])')


if __name__ == '__main__':
    main()