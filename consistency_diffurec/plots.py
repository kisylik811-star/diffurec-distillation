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
def plot_convergence(loss_log_csv, out_path='figures/convergence.pdf'):
    """
    Plot training-loss curves. Expects a CSV with columns
    `epoch,cons_loss,ce_loss` (one row per epoch).
    """
    import csv
    epochs, cons, ce = [], [], []
    with open(loss_log_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row['epoch']))
            cons.append(float(row['cons_loss']))
            ce.append(float(row['ce_loss']))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, cons, color=COLORS['student'], linewidth=2.0)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Consistency loss')
    axes[0].set_title('Consistency loss')

    axes[1].plot(epochs, ce, color=COLORS['baseline'], linewidth=2.0)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Cross-entropy loss')
    axes[1].set_title('Recommendation loss')

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
    args = ap.parse_args()

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

    for name, res in per_dataset.items():
        slug = name.replace('/', '_').replace(' ', '_')
        plot_pareto(res, metric=args.metric,
                    out_path=os.path.join(args.out_dir, f'pareto_{slug}.pdf'))
        plot_latency_bars(res,
                          out_path=os.path.join(args.out_dir, f'latency_{slug}.pdf'))

    print(f'Figures written to {args.out_dir}/')


if __name__ == '__main__':
    main()