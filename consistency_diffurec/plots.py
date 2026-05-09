"""
Plotting utilities for the dissertation.

Black + yellow palette on a white background; serif fonts; consistent style
across figures. All functions take JSON results from `multi_seed_runner.py`
and write PDFs/PNGs to disk.

Run as a script to generate the standard figure pack:

    python plots.py --json_paths results/beauty.json results/toys.json results/ml1m.json \
                    --out_dir figures/

Or import and call individual functions for figures that need extra data
(e.g. sensitivity sweeps, ablation runs).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------- Style ----------
COLORS = {
    'student':    '#1a1a1a',  # black: our distilled student
    'teacher':    '#DAA520',  # goldenrod: teacher (full or truncated)
    'baseline':   '#8B4513',  # saddle brown: alternative baselines
    'tertiary':   '#B8860B',  # dark goldenrod: third series
    'grid':       '#888888',
    'fill_light': '#F4E4A1',  # light yellow for CI fills
}
MARKERS = {'student': 'o', 'teacher': 's', 'baseline': '^', 'tertiary': 'D'}


def apply_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.facecolor':   'white',
        'axes.facecolor':     'white',
        'savefig.facecolor':  'white',
        'font.family':        'serif',
        'font.size':           11,
        'axes.titlesize':      12,
        'axes.labelsize':      11,
        'legend.fontsize':     10,
        'xtick.labelsize':     10,
        'ytick.labelsize':     10,
        'axes.spines.top':     False,
        'axes.spines.right':   False,
        'axes.grid':           True,
        'grid.color':          COLORS['grid'],
        'grid.alpha':          0.3,
        'grid.linestyle':      '--',
        'lines.linewidth':     2.0,
        'lines.markersize':    7,
    })


def _save(fig, out_path):
    Path(os.path.dirname(out_path) or '.').mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def _gather(results, nfe, metric):
    return np.array([
        results['students'][seed][str(nfe)][metric]
        for seed in results['students']
    ])


def _student_mean_std(results, nfe_grid, metric):
    means, stds = [], []
    for nfe in nfe_grid:
        vals = _gather(results, nfe, metric)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(stds)


# ---------- Figure 1: trade-off (NFE vs quality) — main RQ2 figure ----------
def plot_tradeoff_nfe_quality(results_per_dataset, metric='HR@10',
                              out_path='figures/tradeoff_nfe_quality.pdf'):
    """
    Multi-panel: one subplot per dataset.
    Lines: distilled student vs teacher with truncated DDIM.
    Horizontal dashed line: teacher at full NFE.
    """
    apply_style()
    n = len(results_per_dataset)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (ds_name, res) in zip(axes, results_per_dataset.items()):
        nfe_grid = sorted(int(k) for k in res['baseline'].keys())

        # Teacher full reference
        ax.axhline(res['teacher']['full_nfe'][metric], color=COLORS['teacher'],
                   linestyle=':', linewidth=1.8,
                   label=f"Teacher (NFE={res['teacher']['T']})")

        # Truncated DDIM
        truncated = [res['baseline'][str(n_)][metric] for n_ in nfe_grid]
        ax.plot(nfe_grid, truncated,
                marker=MARKERS['baseline'], color=COLORS['baseline'],
                linewidth=2.0, label='Truncated DDIM')

        # Student with error bars
        means, stds = _student_mean_std(res, nfe_grid, metric)
        ax.errorbar(nfe_grid, means, yerr=stds,
                    marker=MARKERS['student'], color=COLORS['student'],
                    linewidth=2.0, capsize=4, label='Distilled student')
        ax.fill_between(nfe_grid, means - stds, means + stds,
                        color=COLORS['fill_light'], alpha=0.4)

        ax.set_xscale('log', base=2)
        ax.set_xticks(nfe_grid)
        ax.set_xticklabels(nfe_grid)
        ax.set_xlabel('NFE (number of forward passes)')
        ax.set_ylabel(metric)
        ax.set_title(ds_name)
        ax.legend(loc='lower right', framealpha=0.95)

    fig.suptitle(f'Quality vs inference cost ({metric})', y=1.02, fontsize=13)
    _save(fig, out_path)


# ---------- Figure 2: Pareto frontier (latency vs quality) ----------
def plot_pareto(results, metric='HR@10', out_path='figures/pareto.pdf'):
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    nfe_grid = sorted(int(k) for k in results['latency']['student'].keys())

    # Student
    s_means, s_stds = _student_mean_std(results, nfe_grid, metric)
    s_lat = [results['latency']['student'][str(n_)] for n_ in nfe_grid]
    ax.errorbar(s_lat, s_means, yerr=s_stds,
                marker=MARKERS['student'], color=COLORS['student'],
                linewidth=2.0, capsize=4, label='Distilled student')
    for nfe, x_, y_ in zip(nfe_grid, s_lat, s_means):
        ax.annotate(f'NFE={nfe}', (x_, y_), textcoords='offset points',
                    xytext=(6, 6), fontsize=9, color=COLORS['student'])

    # Truncated DDIM
    b_q = [results['baseline'][str(n_)][metric] for n_ in nfe_grid]
    b_lat = [results['latency']['teacher_truncated'][str(n_)] for n_ in nfe_grid]
    ax.plot(b_lat, b_q,
            marker=MARKERS['baseline'], color=COLORS['baseline'],
            linewidth=2.0, label='Truncated DDIM')

    # Teacher full
    ax.scatter([results['latency']['teacher_full']],
               [results['teacher']['full_nfe'][metric]],
               marker='*', s=200, color=COLORS['teacher'],
               label=f"Teacher full (NFE={results['teacher']['T']})",
               zorder=5, edgecolors='black', linewidths=0.8)

    ax.set_xscale('log')
    ax.set_xlabel('Inference latency (ms / sample)')
    ax.set_ylabel(metric)
    ax.set_title(f'Pareto: quality vs latency — {results["dataset"]}')
    ax.legend(loc='lower right')
    _save(fig, out_path)


# ---------- Figure 3: latency bar chart ----------
def plot_latency_bars(results, out_path='figures/latency_bars.pdf'):
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    nfe_grid = sorted(int(k) for k in results['latency']['student'].keys())
    x = np.arange(len(nfe_grid))
    w = 0.4

    s_lat = [results['latency']['student'][str(n_)] for n_ in nfe_grid]
    t_lat = [results['latency']['teacher_truncated'][str(n_)] for n_ in nfe_grid]

    ax.bar(x - w/2, t_lat, w, color=COLORS['baseline'], label='Truncated DDIM')
    ax.bar(x + w/2, s_lat, w, color=COLORS['student'], label='Distilled student')
    ax.axhline(results['latency']['teacher_full'], color=COLORS['teacher'],
               linestyle=':', linewidth=1.8,
               label=f"Teacher full (NFE={results['teacher']['T']})")

    ax.set_xticks(x)
    ax.set_xticklabels([f'NFE={n_}' for n_ in nfe_grid])
    ax.set_ylabel('Latency (ms / sample)')
    ax.set_yscale('log')
    ax.set_title(f'Inference latency — {results["dataset"]}')
    ax.legend()
    _save(fig, out_path)


# ---------- Figure 4: speedup factor ----------
def plot_speedup(results_per_dataset, out_path='figures/speedup.pdf'):
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    datasets = list(results_per_dataset.keys())
    nfe_grid = sorted(int(k) for k in next(iter(results_per_dataset.values()))['latency']['student'].keys())
    x = np.arange(len(datasets))
    w = 0.8 / len(nfe_grid)
    cmap = [COLORS['student'], COLORS['tertiary'], COLORS['baseline'], COLORS['teacher']]

    for i, nfe in enumerate(nfe_grid[:4]):  # show first 4 NFEs
        speedups = []
        for ds, res in results_per_dataset.items():
            speedups.append(res['latency']['teacher_full'] /
                            res['latency']['student'][str(nfe)])
        ax.bar(x + (i - len(nfe_grid[:4])/2 + 0.5) * w, speedups, w,
               label=f'NFE={nfe}', color=cmap[i % len(cmap)])

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(r'Speedup vs teacher full ($\times$)')
    ax.set_title('Inference speedup of distilled student')
    ax.legend()
    _save(fig, out_path)


# ---------- Figure 5: sensitivity to a hyperparameter ----------
def plot_sensitivity(param_name, param_values, results_per_value,
                     metric='HR@10', out_path=None):
    """
    `results_per_value`: dict {param_value: results_json} — one full multi-seed
    run per hyperparameter value, all on the same dataset.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    means, stds = [], []
    for v in param_values:
        res = results_per_value[v]
        # use NFE=1 (the most challenging single-step regime)
        vals = _gather(res, 1, metric)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
    means = np.array(means)
    stds = np.array(stds)

    ax.errorbar(param_values, means, yerr=stds,
                marker=MARKERS['student'], color=COLORS['student'],
                linewidth=2.0, capsize=4)
    ax.fill_between(param_values, means - stds, means + stds,
                    color=COLORS['fill_light'], alpha=0.4)

    ax.set_xlabel(param_name)
    ax.set_ylabel(metric)
    ax.set_title(f'Sensitivity to {param_name}')

    out_path = out_path or f'figures/sensitivity_{param_name}.pdf'
    _save(fig, out_path)


# ---------- Figure 6: convergence curves ----------
def plot_convergence(log_dir, out_path='figures/convergence.pdf', skip_warmup=0):
    """
    Plot training loss curves for the student, aggregated across seeds.

    Reads all `seed_*.csv` files in `log_dir`, computes mean ± std across
    seeds for each epoch, and plots a band.

    Parameters
    ----------
    log_dir : str
        Path like `logs/amazon_beauty/` containing per-seed CSV files
        (`seed_1997.csv`, `seed_42.csv`, ...).
    skip_warmup : int
        Skip the first N epochs (loss can be huge at epoch 0 and squashes
        the y-axis).
    """
    import csv
    import glob

    csv_paths = sorted(glob.glob(os.path.join(log_dir, 'seed_*.csv')))
    csv_paths = [p for p in csv_paths if not p.endswith('.val.csv')]
    if not csv_paths:
        print(f'[plot_convergence] no CSVs found in {log_dir}; skip')
        return

    # Collect per-seed series
    series = []
    for p in csv_paths:
        ep, cons, ce, total = [], [], [], []
        with open(p) as f:
            for row in csv.DictReader(f):
                ep.append(int(row['epoch']))
                cons.append(float(row['cons_loss']))
                ce.append(float(row['ce_loss']))
                total.append(float(row['total_loss']))
        series.append((np.array(ep), np.array(cons), np.array(ce), np.array(total)))

    # Pad to common length (early stopping makes seeds finish at different epochs)
    max_len = max(len(s[0]) for s in series)
    def _stack(idx):
        rows = []
        for s in series:
            arr = s[idx]
            if len(arr) < max_len:
                arr = np.concatenate([arr, np.full(max_len - len(arr), np.nan)])
            rows.append(arr)
        return np.vstack(rows)

    epochs = np.arange(max_len)
    cons_mat  = _stack(1)
    ce_mat    = _stack(2)
    total_mat = _stack(3)

    sl = slice(skip_warmup, None)

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, mat, title, color in [
        (axes[0], cons_mat,  'Consistency loss',     COLORS['student']),
        (axes[1], ce_mat,    'Cross-entropy loss',   COLORS['baseline']),
        (axes[2], total_mat, 'Total weighted loss',  COLORS['tertiary']),
    ]:
        m = np.nanmean(mat, axis=0)[sl]
        s = np.nanstd(mat, axis=0, ddof=1)[sl] if mat.shape[0] > 1 else np.zeros_like(m)
        e = epochs[sl]
        ax.plot(e, m, color=color, linewidth=2.0, label='mean across seeds')
        ax.fill_between(e, m - s, m + s, color=COLORS['fill_light'], alpha=0.5,
                        label=r'$\pm$ 1 std')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(f'Student training convergence ({len(series)} seeds)', y=1.02)
    _save(fig, out_path)


def plot_val_curves(log_dir, metric='HR@10', out_path='figures/val_curve.pdf'):
    """
    Validation HR/NDCG over training epochs, aggregated across seeds.
    Reads `seed_*.csv.val.csv` files. Useful for showing convergence of the
    student's recsys quality (not just the loss).
    """
    import csv
    import glob

    csv_paths = sorted(glob.glob(os.path.join(log_dir, 'seed_*.csv.val.csv')))
    if not csv_paths:
        print(f'[plot_val_curves] no val CSVs in {log_dir}; skip')
        return

    series = []
    for p in csv_paths:
        ep, val = [], []
        with open(p) as f:
            for row in csv.DictReader(f):
                ep.append(int(row['epoch']))
                val.append(float(row[metric]))
        series.append((np.array(ep), np.array(val)))

    max_len = max(len(s[0]) for s in series)
    rows = []
    for ep_arr, val_arr in series:
        if len(val_arr) < max_len:
            val_arr = np.concatenate([val_arr, np.full(max_len - len(val_arr), np.nan)])
        rows.append(val_arr)
    mat = np.vstack(rows)
    epochs = series[0][0] if len(series[0][0]) == max_len else np.arange(max_len)

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    m = np.nanmean(mat, axis=0)
    s = np.nanstd(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(m)
    ax.plot(epochs[:len(m)], m, color=COLORS['student'], linewidth=2.0,
            label='mean across seeds')
    ax.fill_between(epochs[:len(m)], m - s, m + s,
                    color=COLORS['fill_light'], alpha=0.5,
                    label=r'$\pm$ 1 std')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Val {metric} (NFE=1)')
    ax.set_title(f'Validation {metric} over training ({len(series)} seeds)')
    ax.legend()
    _save(fig, out_path)


# ---------- Figure: main results bar chart with error bars ----------
def plot_main_results_errorbars(results_per_dataset, metric='HR@10',
                                out_path='figures/main_results_errorbars.pdf'):
    """
    Bar chart per dataset: teacher full / truncated DDIM at NFE=1 / student at NFE=1,
    with std error bars across seeds for the student.
    """
    apply_style()
    datasets = list(results_per_dataset.keys())
    n = len(datasets)
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * n), 4.5))

    x = np.arange(n)
    w = 0.27

    teacher_vals  = [r['teacher']['full_nfe'][metric] for r in results_per_dataset.values()]
    baseline_vals = [r['baseline']['1'][metric] for r in results_per_dataset.values()]
    student_means, student_stds = [], []
    for r in results_per_dataset.values():
        v = _gather(r, 1, metric)
        student_means.append(v.mean())
        student_stds.append(v.std(ddof=1) if len(v) > 1 else 0.0)

    ax.bar(x - w, teacher_vals, w,  color=COLORS['teacher'],  label='Teacher (NFE=T)',
           edgecolor='black', linewidth=0.5)
    ax.bar(x,     baseline_vals, w, color=COLORS['baseline'], label='Truncated DDIM (NFE=1)',
           edgecolor='black', linewidth=0.5)
    ax.bar(x + w, student_means, w, yerr=student_stds, capsize=5,
           color=COLORS['student'], label='Distilled student (NFE=1)',
           edgecolor='black', linewidth=0.5,
           error_kw={'ecolor': COLORS['tertiary'], 'elinewidth': 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(metric)
    ax.set_title(f'Main results ({metric}, mean $\\pm$ std across seeds)')
    ax.legend(loc='best')
    _save(fig, out_path)


# ---------- Figure: bootstrap confidence intervals ----------
def plot_bootstrap_ci(results, metric='HR@10', n_boot=10000,
                      out_path='figures/bootstrap_ci.pdf'):
    """
    For each NFE in the student's grid, draw a bootstrap 95% CI.
    Visually answers: is the student's improvement over truncated-DDIM
    at each NFE statistically meaningful?
    """
    apply_style()
    nfe_grid = sorted(int(k) for k in results['baseline'].keys())

    means, los, his = [], [], []
    base_vals = []
    rng = np.random.default_rng(0)
    for nfe in nfe_grid:
        v = _gather(results, nfe, metric)
        means.append(v.mean())
        if len(v) > 1:
            boots = np.array([rng.choice(v, size=len(v), replace=True).mean()
                              for _ in range(n_boot)])
            los.append(np.percentile(boots, 2.5))
            his.append(np.percentile(boots, 97.5))
        else:
            los.append(v.mean()); his.append(v.mean())
        base_vals.append(results['baseline'][str(nfe)][metric])

    means = np.array(means); los = np.array(los); his = np.array(his)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(nfe_grid, means, marker=MARKERS['student'], color=COLORS['student'],
            linewidth=2.0, label='Student (mean)')
    ax.fill_between(nfe_grid, los, his, color=COLORS['fill_light'], alpha=0.6,
                    label='Student 95% bootstrap CI')
    ax.plot(nfe_grid, base_vals, marker=MARKERS['baseline'], color=COLORS['baseline'],
            linewidth=2.0, linestyle='--', label='Truncated DDIM')
    ax.axhline(results['teacher']['full_nfe'][metric], color=COLORS['teacher'],
               linestyle=':', linewidth=1.8,
               label=f"Teacher full (NFE={results['teacher']['T']})")

    ax.set_xscale('log', base=2)
    ax.set_xticks(nfe_grid)
    ax.set_xticklabels(nfe_grid)
    ax.set_xlabel('NFE')
    ax.set_ylabel(metric)
    ax.set_title(f'95% bootstrap CI — {results["dataset"]}')
    ax.legend()
    _save(fig, out_path)


# ---------- Figure: val vs test generalization gap ----------
def plot_val_vs_test_gap(results, metric='HR@10', nfe_grid=(1, 2, 4, 8, 16, 32),
                         out_path='figures/val_test_gap.pdf'):
    """
    Side-by-side bars comparing val and test metric for the student at each
    NFE, plus reference lines for teacher val and teacher test.

    The point of this figure is to show that the student's generalization
    gap (val - test) tracks the teacher's gap, i.e. distillation does not
    introduce additional overfitting.
    """
    apply_style()

    val_means, val_stds, test_means, test_stds = [], [], [], []
    for nfe in nfe_grid:
        # Test arrays
        test_v = np.array([
            results['students'][s][str(nfe)][metric]
            for s in results['students']
        ])
        # Val arrays — may not exist on older JSONs
        seeds = list(results['students'].keys())
        if '_val' not in results['students'][seeds[0]]:
            print(f'[plot_val_vs_test_gap] no val metrics in JSON for '
                  f'{results["dataset"]}, skipping.')
            return
        val_v = np.array([
            results['students'][s]['_val'][str(nfe)][metric]
            for s in seeds
        ])
        val_means.append(val_v.mean())
        val_stds.append(val_v.std(ddof=1) if len(val_v) > 1 else 0.0)
        test_means.append(test_v.mean())
        test_stds.append(test_v.std(ddof=1) if len(test_v) > 1 else 0.0)

    val_means  = np.array(val_means)
    val_stds   = np.array(val_stds)
    test_means = np.array(test_means)
    test_stds  = np.array(test_stds)

    x = np.arange(len(nfe_grid))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w / 2, val_means, w, yerr=val_stds, capsize=3,
           color=COLORS['student'], alpha=0.85,
           edgecolor='black', linewidth=0.5,
           label='Student — Validation')
    ax.bar(x + w / 2, test_means, w, yerr=test_stds, capsize=3,
           color=COLORS['fill_light'], alpha=0.95,
           edgecolor='black', linewidth=0.5,
           label='Student — Test')

    # Teacher reference lines
    teacher_test = results['teacher']['full_nfe'].get(metric)
    teacher_val  = results['teacher'].get('full_nfe_val', {}).get(metric)
    if teacher_test is not None:
        ax.axhline(teacher_test, color=COLORS['teacher'],
                   linestyle=':', linewidth=1.8,
                   label=f'Teacher — Test ({teacher_test:.2f})')
    if teacher_val is not None:
        ax.axhline(teacher_val, color=COLORS['baseline'],
                   linestyle='--', linewidth=1.5, alpha=0.8,
                   label=f'Teacher — Validation ({teacher_val:.2f})')

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nfe_grid])
    ax.set_xlabel('NFE')
    ax.set_ylabel(metric)
    ax.set_title(f'Validation vs test {metric} — {results["dataset"]}\n'
                 f'(generalization gap check)', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    _save(fig, out_path)


# ---------- Figure 7: ablation bar chart ----------
def plot_ablation(ablation_results, metric='HR@10', out_path='figures/ablation.pdf'):
    """
    `ablation_results`: dict {variant_name: results_json}.
    Plots NFE=1 metric for each variant with error bars.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    names = list(ablation_results.keys())
    means, stds = [], []
    for name in names:
        vals = _gather(ablation_results[name], 1, metric)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    x = np.arange(len(names))
    colors = [COLORS['student'] if n.startswith('Full') else COLORS['baseline']
              for n in names]
    ax.bar(x, means, yerr=stds, color=colors, capsize=4,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel(metric)
    ax.set_title('Ablation study (NFE = 1)')
    _save(fig, out_path)


# ---------- Standalone runner ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json_paths', nargs='+', required=True)
    ap.add_argument('--out_dir', default='figures')
    ap.add_argument('--metric', default='HR@10')
    ap.add_argument('--logs_root', default='logs',
                    help='Root dir of per-dataset, per-seed CSV logs '
                         '(default: logs/, written by multi_seed_runner.py).')
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    per_dataset = {}
    for path in args.json_paths:
        with open(path) as f:
            r = json.load(f)
        per_dataset[r['dataset']] = r

    plot_tradeoff_nfe_quality(per_dataset, metric=args.metric,
                              out_path=os.path.join(args.out_dir, 'tradeoff_HR10.pdf'))
    plot_tradeoff_nfe_quality(per_dataset, metric='NDCG@10',
                              out_path=os.path.join(args.out_dir, 'tradeoff_NDCG10.pdf'))
    plot_speedup(per_dataset, out_path=os.path.join(args.out_dir, 'speedup.pdf'))
    plot_main_results_errorbars(per_dataset, metric=args.metric,
                                out_path=os.path.join(args.out_dir, 'main_results_errorbars.pdf'))

    for name, res in per_dataset.items():
        slug = name.replace('/', '_').replace(' ', '_')
        plot_pareto(res, metric=args.metric,
                    out_path=os.path.join(args.out_dir, f'pareto_{slug}.pdf'))
        plot_latency_bars(res,
                          out_path=os.path.join(args.out_dir, f'latency_{slug}.pdf'))
        plot_bootstrap_ci(res, metric=args.metric,
                          out_path=os.path.join(args.out_dir, f'bootstrap_ci_{slug}.pdf'))

        # Convergence + val curves require per-seed CSVs from multi_seed_runner.
        log_dir = os.path.join(args.logs_root, name)
        if os.path.isdir(log_dir):
            plot_convergence(log_dir,
                             out_path=os.path.join(args.out_dir, f'convergence_{slug}.pdf'))
            plot_val_curves(log_dir, metric=args.metric,
                            out_path=os.path.join(args.out_dir, f'val_curve_{slug}.pdf'))
        else:
            print(f'[main] {log_dir} not found; skipping convergence plots for {name}')

    print(f'Figures written to {args.out_dir}/')


if __name__ == '__main__':
    main()