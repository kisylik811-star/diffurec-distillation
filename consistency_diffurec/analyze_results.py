"""
Post-hoc analysis for RCCD experiments.

Reads artifacts from artifacts/<dataset>/<run_name>/ and produces:
  1. Sensitivity grid: HR@10 / NDCG@10 vs (beta, tau)  — table + heatmap
  2. Length-aware analysis: HR@10 by sequence length bin per run
  3. Multi-seed statistics: mean ± std, paired Wilcoxon, bootstrap CI
  4. Latency Pareto: quality vs ms/sample

Run as:
    python analyze_results.py --dataset toys --mode sensitivity
    python analyze_results.py --dataset toys --mode length_aware --run_name seed1997_beta0.5_tau0.1
    python analyze_results.py --dataset toys --mode multi_seed
    python analyze_results.py --dataset toys --mode all
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    from scipy import stats as sst
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


COLORS = {
    'student':   '#1a1a1a',
    'teacher':   '#DAA520',
    'baseline':  '#8B4513',
    'fill':      '#F4E4A1',
}


def _apply_style():
    if not HAS_PLT:
        return
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'font.family': 'serif',
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 2.0,
    })


def _save(fig, out_path):
    Path(os.path.dirname(out_path) or '.').mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def load_all_runs(dataset, artifacts_root='artifacts'):
    """Returns list of dicts, one per run, with config + summary loaded."""
    runs = []
    pattern = os.path.join(artifacts_root, dataset, '*', 'summary.json')
    for path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(path)
        with open(path) as f:
            summary = json.load(f)
        with open(os.path.join(run_dir, 'config.json')) as f:
            config = json.load(f)
        summary['run_dir'] = run_dir
        summary['config'] = config
        runs.append(summary)
    return runs


def load_teacher_reference(dataset, artifacts_root='artifacts'):
    path = os.path.join(artifacts_root, dataset, 'teacher', 'reference.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================
# Analysis 1: Sensitivity grid
# ============================================================
def sensitivity_analysis(dataset, metric='HR@10', nfe='1',
                        artifacts_root='artifacts',
                        out_dir='figures', seed_filter=None):
    """Build a (beta, tau) grid of metric values from single-seed runs."""
    runs = load_all_runs(dataset, artifacts_root)
    if seed_filter is not None:
        runs = [r for r in runs if r['random_seed'] == seed_filter]

    # Collect all (beta, tau) -> metric
    grid = {}
    for r in runs:
        beta = r['contrast_weight']
        tau = r['contrast_temperature']
        val = r['test_metrics_per_nfe'][nfe][metric]
        grid[(beta, tau)] = val

    betas = sorted({b for b, _ in grid})
    taus = sorted({t for _, t in grid})

    # Build matrix
    matrix = np.full((len(betas), len(taus)), np.nan)
    for i, b in enumerate(betas):
        for j, t in enumerate(taus):
            if (b, t) in grid:
                matrix[i, j] = grid[(b, t)]

    # ---- Print table ----
    print(f'\n=== Sensitivity grid for {dataset}, {metric}@NFE={nfe} ===')
    header = '          ' + '  '.join(f'tau={t:<6}' for t in taus)
    print(header)
    for i, b in enumerate(betas):
        row = f'beta={b:<5} '
        for j in range(len(taus)):
            v = matrix[i, j]
            row += f'{v:>9.4f}  ' if not np.isnan(v) else '    --     '
        print(row)

    # Find best
    if not np.all(np.isnan(matrix)):
        best_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
        best_beta = betas[best_idx[0]]
        best_tau = taus[best_idx[1]]
        best_val = matrix[best_idx]
        print(f'\nBest: beta={best_beta}, tau={best_tau} -> {metric}={best_val:.4f}')

    # ---- Heatmap ----
    if HAS_PLT:
        _apply_style()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(matrix, aspect='auto', cmap='cividis')
        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels([f'{t}' for t in taus])
        ax.set_yticks(range(len(betas)))
        ax.set_yticklabels([f'{b}' for b in betas])
        ax.set_xlabel('Contrast temperature τ')
        ax.set_ylabel('Contrast weight β')
        ax.set_title(f'{metric} @ NFE={nfe} ({dataset})')

        # Annotate cells
        for i in range(len(betas)):
            for j in range(len(taus)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            color='white' if v < np.nanmean(matrix) else 'black',
                            fontsize=10)

        plt.colorbar(im, ax=ax)
        _save(fig, os.path.join(out_dir, f'sensitivity_{dataset}_{metric.replace("@","")}.pdf'))

    return matrix, betas, taus


# ============================================================
# Analysis 2: Length-aware
# ============================================================
def length_aware_analysis(dataset, run_name, artifacts_root='artifacts',
                         out_dir='figures', n_bins=5):
    """HR@10 broken down by history length quintiles."""
    run_dir = os.path.join(artifacts_root, dataset, run_name)
    pred_path = os.path.join(run_dir, 'test_predictions_nfe1.npz')
    if not os.path.exists(pred_path):
        print(f'[length-aware] no predictions at {pred_path}')
        return None

    data = np.load(pred_path)
    lengths = data['hist_lengths']
    ks = data['ks']
    hits = data['hit_at_k']  # (N, len(ks))

    # Index of HR@10 column
    try:
        k10_idx = list(ks).index(10)
    except ValueError:
        k10_idx = 1  # fallback

    # Quantile bins
    quantiles = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    bin_results = []
    for i in range(n_bins):
        lo, hi = quantiles[i], quantiles[i + 1]
        if i == n_bins - 1:
            mask = (lengths >= lo) & (lengths <= hi)
        else:
            mask = (lengths >= lo) & (lengths < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        hit_rate = float(hits[mask, k10_idx].mean()) * 100
        # Wilson 95% CI
        k = int(hits[mask, k10_idx].sum())
        if n > 0:
            p = k / n
            z = 1.96
            denom = 1 + z*z/n
            center = (p + z*z/(2*n)) / denom
            spread = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
            ci_lo, ci_hi = max(0, center - spread)*100, min(1, center + spread)*100
        else:
            ci_lo, ci_hi = 0, 0
        bin_results.append({
            'bin': i+1, 'len_lo': int(lo), 'len_hi': int(hi),
            'n': n, 'HR@10': hit_rate, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        })

    # Print
    print(f'\n=== Length-aware analysis: {dataset} / {run_name} ===')
    print(f'{"Bin":<5} {"Range":<15} {"n":<6} {"HR@10":<8} {"95% Wilson CI":<20}')
    for r in bin_results:
        print(f'{r["bin"]:<5} {r["len_lo"]}-{r["len_hi"]:<10} {r["n"]:<6} '
              f'{r["HR@10"]:<8.2f} [{r["ci_lo"]:.2f}, {r["ci_hi"]:.2f}]')

    # Plot
    if HAS_PLT and bin_results:
        _apply_style()
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [f'{r["len_lo"]}-{r["len_hi"]}' for r in bin_results]
        ys = [r['HR@10'] for r in bin_results]
        ylo = [r['HR@10'] - r['ci_lo'] for r in bin_results]
        yhi = [r['ci_hi'] - r['HR@10'] for r in bin_results]
        ax.bar(xs, ys, yerr=[ylo, yhi], capsize=4,
               color=COLORS['student'], edgecolor='black', linewidth=0.5)
        ax.set_xlabel('History length range')
        ax.set_ylabel('HR@10 (%)')
        ax.set_title(f'Performance by history length — {run_name}')
        _save(fig, os.path.join(out_dir, f'length_aware_{dataset}_{run_name}.pdf'))

    return bin_results


def length_aware_compare(dataset, run_names, artifacts_root='artifacts',
                        out_dir='figures', n_bins=5, labels=None):
    """Side-by-side length-aware comparison of multiple runs."""
    all_results = {}
    for rn in run_names:
        res = length_aware_analysis(dataset, rn, artifacts_root,
                                    out_dir=out_dir, n_bins=n_bins)
        if res is not None:
            all_results[rn] = res

    if not HAS_PLT or len(all_results) < 2:
        return all_results

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    runs_list = list(all_results.keys())
    bins_x = [f'{r["len_lo"]}-{r["len_hi"]}' for r in all_results[runs_list[0]]]
    width = 0.8 / len(runs_list)
    x = np.arange(len(bins_x))

    palette = [COLORS['baseline'], COLORS['student'], COLORS['teacher']]
    for i, rn in enumerate(runs_list):
        ys = [r['HR@10'] for r in all_results[rn]]
        label = labels[i] if labels else rn
        ax.bar(x + (i - len(runs_list)/2 + 0.5)*width, ys, width,
               color=palette[i % len(palette)], edgecolor='black',
               linewidth=0.5, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(bins_x)
    ax.set_xlabel('History length range')
    ax.set_ylabel('HR@10 (%)')
    ax.set_title(f'Length-aware comparison ({dataset})')
    ax.legend()
    _save(fig, os.path.join(out_dir, f'length_aware_compare_{dataset}.pdf'))
    return all_results


# ============================================================
# Analysis 3: Multi-seed statistics
# ============================================================
def multi_seed_statistics(dataset, beta_target, tau_target,
                         beta_baseline=0.0, tau_baseline=0.1,
                         metric='HR@10', nfe='1',
                         artifacts_root='artifacts', n_boot=10000):
    """Compare RCCD (beta_target, tau_target) vs CD baseline across seeds."""
    runs = load_all_runs(dataset, artifacts_root)

    rccd_vals = [r['test_metrics_per_nfe'][nfe][metric]
                 for r in runs
                 if abs(r['contrast_weight'] - beta_target) < 1e-9
                 and abs(r['contrast_temperature'] - tau_target) < 1e-9]
    base_vals = [r['test_metrics_per_nfe'][nfe][metric]
                 for r in runs
                 if abs(r['contrast_weight'] - beta_baseline) < 1e-9
                 and abs(r['contrast_temperature'] - tau_baseline) < 1e-9]

    print(f'\n=== Multi-seed stats: {dataset}, {metric}@NFE={nfe} ===')
    print(f'RCCD (β={beta_target}, τ={tau_target}): n={len(rccd_vals)}, '
          f'values={rccd_vals}')
    print(f'Baseline (β=0):                          n={len(base_vals)}, '
          f'values={base_vals}')

    if len(rccd_vals) == 0 or len(base_vals) == 0:
        print('Not enough runs.')
        return None

    rccd_arr = np.array(rccd_vals)
    base_arr = np.array(base_vals)
    rccd_mean, rccd_std = rccd_arr.mean(), rccd_arr.std(ddof=1) if len(rccd_arr) > 1 else 0
    base_mean, base_std = base_arr.mean(), base_arr.std(ddof=1) if len(base_arr) > 1 else 0
    diff = rccd_mean - base_mean

    print(f'\nRCCD:     {rccd_mean:.4f} ± {rccd_std:.4f}')
    print(f'Baseline: {base_mean:.4f} ± {base_std:.4f}')
    print(f'Diff:     {diff:+.4f}')

    # Paired Wilcoxon if same seeds available
    if HAS_SCIPY and len(rccd_arr) == len(base_arr) and len(rccd_arr) >= 3:
        try:
            stat, p = sst.wilcoxon(rccd_arr, base_arr)
            print(f'Paired Wilcoxon: stat={stat:.3f}, p={p:.4f}')
        except ValueError as e:
            print(f'Wilcoxon could not run: {e}')

    # Bootstrap CI on the difference
    if len(rccd_arr) >= 2 and len(base_arr) >= 2:
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(n_boot):
            r_sample = rng.choice(rccd_arr, len(rccd_arr), replace=True).mean()
            b_sample = rng.choice(base_arr, len(base_arr), replace=True).mean()
            diffs.append(r_sample - b_sample)
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f'Bootstrap 95% CI on diff: [{lo:+.4f}, {hi:+.4f}]')
        print(f'P(RCCD > baseline) = {(diffs > 0).mean():.3f}')

    return {
        'rccd_mean': float(rccd_mean), 'rccd_std': float(rccd_std),
        'baseline_mean': float(base_mean), 'baseline_std': float(base_std),
        'diff': float(diff),
    }


# ============================================================
# Analysis 4: Latency Pareto
# ============================================================
def latency_pareto(dataset, best_run_name=None, baseline_run_name=None,
                  artifacts_root='artifacts', out_dir='figures',
                  metric='HR@10'):
    teacher_ref = load_teacher_reference(dataset, artifacts_root)
    runs = load_all_runs(dataset, artifacts_root)

    points = []
    if teacher_ref:
        points.append(('Teacher (NFE=T)',
                       teacher_ref['latency_ms'],
                       teacher_ref['metrics_full_nfe'][metric],
                       COLORS['teacher'], '*'))

    target_runs = []
    if best_run_name:
        target_runs.append((best_run_name, 'RCCD', COLORS['student'], 'o'))
    if baseline_run_name:
        target_runs.append((baseline_run_name, 'CD baseline', COLORS['baseline'], 's'))

    for run_name, label, color, marker in target_runs:
        r = next((x for x in runs if x['run_name'] == run_name), None)
        if r is None:
            continue
        for nfe_str, lat_ms in r['latency_per_nfe_ms'].items():
            m = r['test_metrics_per_nfe'][nfe_str][metric]
            points.append((f'{label} NFE={nfe_str}', lat_ms, m, color, marker))

    if not HAS_PLT or not points:
        return points

    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, lat, m, color, marker in points:
        ax.scatter(lat, m, s=120, color=color, marker=marker,
                   edgecolor='black', linewidth=0.6, zorder=3)
        ax.annotate(name, (lat, m), textcoords='offset points',
                    xytext=(6, 6), fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Inference latency (ms / sample, log scale)')
    ax.set_ylabel(metric)
    ax.set_title(f'Quality vs latency — {dataset}')
    _save(fig, os.path.join(out_dir, f'pareto_{dataset}.pdf'))
    return points


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--mode', choices=['sensitivity', 'length_aware',
                                       'length_compare', 'multi_seed',
                                       'pareto', 'all'],
                   default='all')
    p.add_argument('--artifacts_root', default='artifacts')
    p.add_argument('--out_dir', default='figures')
    p.add_argument('--run_name', default=None,
                   help='For length_aware mode.')
    p.add_argument('--metric', default='HR@10')
    p.add_argument('--nfe', default='1')
    p.add_argument('--seed_filter', type=int, default=None,
                   help='Restrict sensitivity to one seed.')
    p.add_argument('--beta_target', type=float, default=0.5)
    p.add_argument('--tau_target', type=float, default=0.1)
    args = p.parse_args()

    if args.mode in ('sensitivity', 'all'):
        for metric in ['HR@10', 'NDCG@10']:
            sensitivity_analysis(args.dataset, metric=metric, nfe=args.nfe,
                               artifacts_root=args.artifacts_root,
                               out_dir=args.out_dir,
                               seed_filter=args.seed_filter)

    if args.mode == 'length_aware' and args.run_name:
        length_aware_analysis(args.dataset, args.run_name,
                            artifacts_root=args.artifacts_root,
                            out_dir=args.out_dir)

    if args.mode == 'length_compare':
        # auto-pick: best RCCD + baseline
        baseline_rn = f'seed1997_beta0.0_tau0.1'
        best_rn = f'seed1997_beta{args.beta_target}_tau{args.tau_target}'
        length_aware_compare(args.dataset, [baseline_rn, best_rn],
                            artifacts_root=args.artifacts_root,
                            out_dir=args.out_dir,
                            labels=['CD baseline', 'RCCD'])

    if args.mode in ('multi_seed', 'all'):
        multi_seed_statistics(args.dataset,
                            beta_target=args.beta_target,
                            tau_target=args.tau_target,
                            metric=args.metric, nfe=args.nfe,
                            artifacts_root=args.artifacts_root)

    if args.mode in ('pareto', 'all'):
        baseline_rn = f'seed1997_beta0.0_tau0.1'
        best_rn = f'seed1997_beta{args.beta_target}_tau{args.tau_target}'
        latency_pareto(args.dataset, best_run_name=best_rn,
                      baseline_run_name=baseline_rn,
                      artifacts_root=args.artifacts_root,
                      out_dir=args.out_dir, metric=args.metric)


if __name__ == '__main__':
    main()