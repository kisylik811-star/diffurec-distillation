"""
Plotting utilities for the dissertation.

Black + yellow palette on a white background; serif fonts; consistent style
across figures. All functions take JSON results from `multi_seed_runner.py`
(new multi-variant layout) and write PDFs/PNGs to disk.

The new JSON layout supports multiple variants:
    results['variants'][variant_name][seed][nfe][metric]
    results['latency']['student'][variant_name][nfe]

Most plots in this file pick one "headline" variant (default: 'full_racd')
and treat the rest as comparisons (e.g. ablation in `plot_ablation_bars`).

Run as a script for the standard figure pack:

    python plots.py --json_paths results/beauty.json results/toys.json \
                    --out_dir figures/ --headline full_racd
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------- Style ----------
COLORS = {
    'student':    '#1a1a1a',  # black: distilled student / RACD
    'teacher':    '#DAA520',  # goldenrod: teacher (full or truncated)
    'baseline':   '#8B4513',  # saddle brown: alternative baselines (incl. Vanilla CD)
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


# --------------------------------------------------------------------- #
#  Variant-aware accessors                                              #
# --------------------------------------------------------------------- #
def _seeds_dict(results, variant):
    """Return seed -> {nfe -> {metric -> float}} for a variant."""
    if 'variants' in results:
        return results['variants'][variant]
    # legacy single-variant layout
    return results['students']


def _gather(results, variant, nfe, metric):
    sd = _seeds_dict(results, variant)
    return np.array([sd[seed][str(nfe)][metric] for seed in sd])


def _mean_std(results, variant, nfe_grid, metric):
    means, stds = [], []
    for nfe in nfe_grid:
        vals = _gather(results, variant, nfe, metric)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(stds)


def _student_latency(results, variant):
    """Return {nfe_str -> ms}."""
    lat = results['latency']['student']
    if variant in lat:
        return lat[variant]
    return lat  # legacy flat


# --------------------------------------------------------------------- #
#  Figure 1: NFE vs quality (main RQ figure)                            #
# --------------------------------------------------------------------- #
def plot_tradeoff_nfe_quality(results_per_dataset, headline='full_racd',
                              compare='vanilla_cd', metric='HR@10',
                              out_path='figures/tradeoff_nfe_quality.pdf'):
    """
    Multi-panel: one subplot per dataset.
    Lines: headline variant (e.g. RACD), compare variant (Vanilla CD),
    truncated DDIM. Horizontal dashed line: teacher full NFE.
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
                linewidth=1.8, linestyle='--', label='Truncated DDIM')

        # Compare variant (Vanilla CD)
        if compare and compare in res.get('variants', {}):
            cmp_grid = sorted(int(k) for k in
                              next(iter(_seeds_dict(res, compare).values())).keys())
            cmp_means, cmp_stds = _mean_std(res, compare, cmp_grid, metric)
            ax.errorbar(cmp_grid, cmp_means, yerr=cmp_stds,
                        marker=MARKERS['tertiary'], color=COLORS['tertiary'],
                        linewidth=1.8, capsize=3, label=compare.replace('_', ' '))

        # Headline variant (RACD)
        if headline in res.get('variants', {}):
            head_grid = sorted(int(k) for k in
                               next(iter(_seeds_dict(res, headline).values())).keys())
            means, stds = _mean_std(res, headline, head_grid, metric)
            ax.errorbar(head_grid, means, yerr=stds,
                        marker=MARKERS['student'], color=COLORS['student'],
                        linewidth=2.2, capsize=4, label=headline.replace('_', ' '))
            ax.fill_between(head_grid, means - stds, means + stds,
                            color=COLORS['fill_light'], alpha=0.4)

        ax.set_xscale('log', base=2)
        ax.set_xticks(nfe_grid)
        ax.set_xticklabels(nfe_grid)
        ax.set_xlabel('NFE')
        ax.set_ylabel(metric)
        ax.set_title(ds_name)
        ax.legend(loc='lower right', framealpha=0.95)

    fig.suptitle(f'Quality vs inference cost ({metric})', y=1.02, fontsize=13)
    _save(fig, out_path)


# --------------------------------------------------------------------- #
#  Figure 2: Pareto frontier (latency vs quality)                       #
# --------------------------------------------------------------------- #
def plot_pareto(results, headline='full_racd', compare='vanilla_cd',
                metric='HR@10', out_path='figures/pareto.pdf'):
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    head_lat = _student_latency(results, headline)
    nfe_grid = sorted(int(k) for k in head_lat.keys())

    # Headline
    s_means, s_stds = _mean_std(results, headline, nfe_grid, metric)
    s_lat = [head_lat[str(n_)] for n_ in nfe_grid]
    ax.errorbar(s_lat, s_means, yerr=s_stds,
                marker=MARKERS['student'], color=COLORS['student'],
                linewidth=2.2, capsize=4, label=headline.replace('_', ' '))
    for nfe, x_, y_ in zip(nfe_grid, s_lat, s_means):
        ax.annotate(f'NFE={nfe}', (x_, y_), textcoords='offset points',
                    xytext=(6, 6), fontsize=9, color=COLORS['student'])

    # Compare variant
    if compare and compare in results.get('variants', {}):
        c_lat = _student_latency(results, compare)
        c_grid = sorted(int(k) for k in c_lat.keys())
        c_means, _ = _mean_std(results, compare, c_grid, metric)
        c_lat_vals = [c_lat[str(n_)] for n_ in c_grid]
        ax.plot(c_lat_vals, c_means,
                marker=MARKERS['tertiary'], color=COLORS['tertiary'],
                linewidth=1.8, linestyle='--', label=compare.replace('_', ' '))

    # Truncated DDIM
    b_q = [results['baseline'][str(n_)][metric] for n_ in nfe_grid]
    b_lat = [results['latency']['teacher_truncated'][str(n_)] for n_ in nfe_grid]
    ax.plot(b_lat, b_q,
            marker=MARKERS['baseline'], color=COLORS['baseline'],
            linewidth=1.8, label='Truncated DDIM')

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


# --------------------------------------------------------------------- #
#  Figure 3: latency bars                                               #
# --------------------------------------------------------------------- #
def plot_latency_bars(results, headline='full_racd', out_path='figures/latency_bars.pdf'):
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    head_lat = _student_latency(results, headline)
    nfe_grid = sorted(int(k) for k in head_lat.keys())
    x = np.arange(len(nfe_grid))
    w = 0.4

    s_lat = [head_lat[str(n_)] for n_ in nfe_grid]
    t_lat = [results['latency']['teacher_truncated'][str(n_)] for n_ in nfe_grid]

    ax.bar(x - w/2, t_lat, w, color=COLORS['baseline'], label='Truncated DDIM')
    ax.bar(x + w/2, s_lat, w, color=COLORS['student'],
           label=headline.replace('_', ' '))
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


# --------------------------------------------------------------------- #
#  Figure 4: speedup factor across datasets                             #
# --------------------------------------------------------------------- #
def plot_speedup(results_per_dataset, headline='full_racd',
                 out_path='figures/speedup.pdf'):
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    datasets = list(results_per_dataset.keys())
    first = next(iter(results_per_dataset.values()))
    nfe_grid = sorted(int(k) for k in _student_latency(first, headline).keys())
    x = np.arange(len(datasets))
    w = 0.8 / max(len(nfe_grid[:4]), 1)
    cmap = [COLORS['student'], COLORS['tertiary'], COLORS['baseline'], COLORS['teacher']]

    for i, nfe in enumerate(nfe_grid[:4]):
        speedups = []
        for ds, res in results_per_dataset.items():
            lat = _student_latency(res, headline)
            speedups.append(res['latency']['teacher_full'] / lat[str(nfe)])
        ax.bar(x + (i - len(nfe_grid[:4])/2 + 0.5) * w, speedups, w,
               label=f'NFE={nfe}', color=cmap[i % len(cmap)])

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(r'Speedup vs teacher full ($\times$)')
    ax.set_title(f'Inference speedup ({headline.replace("_", " ")})')
    ax.legend()
    _save(fig, out_path)


# --------------------------------------------------------------------- #
#  Figure 5: hyperparameter sensitivity (1 seed per point typical)      #
# --------------------------------------------------------------------- #
def plot_sensitivity(param_name, param_values, results_per_value,
                     headline='full_racd', metric='HR@10', out_path=None):
    """
    `results_per_value`: dict {param_value: results_json}.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    means, stds = [], []
    for v in param_values:
        res = results_per_value[v]
        vals = _gather(res, headline, 1, metric)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
    means = np.array(means); stds = np.array(stds)

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


# --------------------------------------------------------------------- #
#  Figure 6: main results bar chart                                     #
# --------------------------------------------------------------------- #
def plot_main_results_errorbars(results_per_dataset, headline='full_racd',
                                compare='vanilla_cd', metric='HR@10',
                                out_path='figures/main_results_errorbars.pdf'):
    apply_style()
    datasets = list(results_per_dataset.keys())
    n = len(datasets)
    fig, ax = plt.subplots(figsize=(max(6, 2.8 * n), 4.5))

    x = np.arange(n)
    w = 0.22

    teacher_vals  = [r['teacher']['full_nfe'][metric] for r in results_per_dataset.values()]
    baseline_vals = [r['baseline']['1'][metric] for r in results_per_dataset.values()]
    cmp_means, cmp_stds = [], []
    head_means, head_stds = [], []
    for r in results_per_dataset.values():
        if compare in r.get('variants', {}):
            v = _gather(r, compare, 1, metric)
            cmp_means.append(v.mean()); cmp_stds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        else:
            cmp_means.append(np.nan); cmp_stds.append(0.0)
        v = _gather(r, headline, 1, metric)
        head_means.append(v.mean()); head_stds.append(v.std(ddof=1) if len(v) > 1 else 0.0)

    ax.bar(x - 1.5*w, teacher_vals, w,  color=COLORS['teacher'],
           label=f'Teacher (NFE=T)', edgecolor='black', linewidth=0.5)
    ax.bar(x - 0.5*w, baseline_vals, w, color=COLORS['baseline'],
           label='Truncated DDIM (NFE=1)', edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.5*w, cmp_means, w, yerr=cmp_stds, capsize=4,
           color=COLORS['tertiary'], label=f'{compare.replace("_", " ")} (NFE=1)',
           edgecolor='black', linewidth=0.5,
           error_kw={'ecolor': 'black', 'elinewidth': 0.8})
    ax.bar(x + 1.5*w, head_means, w, yerr=head_stds, capsize=4,
           color=COLORS['student'], label=f'{headline.replace("_", " ")} (NFE=1)',
           edgecolor='black', linewidth=0.5,
           error_kw={'ecolor': COLORS['tertiary'], 'elinewidth': 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(metric)
    ax.set_title(f'Main results ({metric}, mean $\\pm$ std across seeds)')
    ax.legend(loc='best')
    _save(fig, out_path)


# --------------------------------------------------------------------- #
#  Figure 7: bootstrap CI                                               #
# --------------------------------------------------------------------- #
def plot_bootstrap_ci(results, headline='full_racd', metric='HR@10',
                      n_boot=10000, out_path='figures/bootstrap_ci.pdf'):
    apply_style()
    nfe_grid = sorted(int(k) for k in results['baseline'].keys())

    means, los, his, base_vals = [], [], [], []
    rng = np.random.default_rng(0)
    for nfe in nfe_grid:
        v = _gather(results, headline, nfe, metric)
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
            linewidth=2.0, label=f'{headline.replace("_", " ")} (mean)')
    ax.fill_between(nfe_grid, los, his, color=COLORS['fill_light'], alpha=0.6,
                    label='95% bootstrap CI')
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


# --------------------------------------------------------------------- #
#  Figure 8: ablation bars (Block 1 / Block 2)                          #
# --------------------------------------------------------------------- #
def plot_ablation_bars(results, variants, baseline_variant, metric='HR@10',
                       nfe=1, title=None, out_path='figures/ablation.pdf'):
    """
    Bar chart for an ablation block: each bar is one variant, error bars are
    std over seeds. The baseline_variant is highlighted in a contrasting
    colour to make the reference obvious.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    means, stds = [], []
    for v in variants:
        vals = _gather(results, v, nfe, metric)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    x = np.arange(len(variants))
    colors = [COLORS['baseline'] if v == baseline_variant else COLORS['student']
              for v in variants]
    ax.bar(x, means, yerr=stds, color=colors, capsize=4,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', ' ') for v in variants],
                       rotation=20, ha='right')
    ax.set_ylabel(f'{metric} (NFE={nfe})')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Ablation (NFE={nfe}, baseline = {baseline_variant})')
    _save(fig, out_path)


# --------------------------------------------------------------------- #
#  Convergence + val curves (per-seed CSVs from logs)                   #
# --------------------------------------------------------------------- #
def plot_convergence(log_dir, out_path='figures/convergence.pdf', skip_warmup=0):
    import csv, glob
    csv_paths = sorted(glob.glob(os.path.join(log_dir, 'seed_*.csv')))
    csv_paths = [p for p in csv_paths if not p.endswith('.val.csv')]
    if not csv_paths:
        print(f'[plot_convergence] no CSVs found in {log_dir}; skip')
        return

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
    cons_mat  = _stack(1); ce_mat = _stack(2); total_mat = _stack(3)
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
        ax.set_xlabel('Epoch'); ax.set_ylabel(title); ax.set_title(title)
        ax.legend(loc='upper right', fontsize=9)
    fig.suptitle(f'Student training convergence ({len(series)} seeds)', y=1.02)
    _save(fig, out_path)


def plot_val_curves(log_dir, metric='HR@10', out_path='figures/val_curve.pdf'):
    import csv, glob
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
    ax.set_xlabel('Epoch'); ax.set_ylabel(f'Val {metric} (NFE=1)')
    ax.set_title(f'Validation {metric} over training ({len(series)} seeds)')
    ax.legend()
    _save(fig, out_path)


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json_paths', nargs='+', required=True)
    ap.add_argument('--out_dir', default='figures')
    ap.add_argument('--metric', default='HR@10')
    ap.add_argument('--headline', default='full_racd')
    ap.add_argument('--compare', default='vanilla_cd')
    ap.add_argument('--logs_root', default='logs')
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    per_dataset = {}
    for path in args.json_paths:
        with open(path) as f:
            r = json.load(f)
        per_dataset[r['dataset']] = r

    plot_tradeoff_nfe_quality(per_dataset, headline=args.headline,
                              compare=args.compare, metric=args.metric,
                              out_path=os.path.join(args.out_dir, 'tradeoff_HR10.pdf'))
    plot_tradeoff_nfe_quality(per_dataset, headline=args.headline,
                              compare=args.compare, metric='NDCG@10',
                              out_path=os.path.join(args.out_dir, 'tradeoff_NDCG10.pdf'))
    plot_speedup(per_dataset, headline=args.headline,
                 out_path=os.path.join(args.out_dir, 'speedup.pdf'))
    plot_main_results_errorbars(per_dataset, headline=args.headline,
                                compare=args.compare, metric=args.metric,
                                out_path=os.path.join(args.out_dir, 'main_results_errorbars.pdf'))

    for name, res in per_dataset.items():
        slug = name.replace('/', '_').replace(' ', '_')
        plot_pareto(res, headline=args.headline, compare=args.compare,
                    metric=args.metric,
                    out_path=os.path.join(args.out_dir, f'pareto_{slug}.pdf'))
        plot_latency_bars(res, headline=args.headline,
                          out_path=os.path.join(args.out_dir, f'latency_{slug}.pdf'))
        plot_bootstrap_ci(res, headline=args.headline, metric=args.metric,
                          out_path=os.path.join(args.out_dir, f'bootstrap_ci_{slug}.pdf'))

        log_dir = os.path.join(args.logs_root, name)
        if os.path.isdir(log_dir):
            plot_convergence(log_dir,
                             out_path=os.path.join(args.out_dir, f'convergence_{slug}.pdf'))
            plot_val_curves(log_dir, metric=args.metric,
                            out_path=os.path.join(args.out_dir, f'val_curve_{slug}.pdf'))

    print(f'Figures written to {args.out_dir}/')


if __name__ == '__main__':
    main()