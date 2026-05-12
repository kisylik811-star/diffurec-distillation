"""
Unified analysis script.

Consolidates all post-training analysis into one file:
  - statistical aggregation (from former statistics.py)
  - main multi-dataset plots (from former plots.py)
  - model-aware plots that need loaded checkpoints (from former analysis_plots.py)
  - sweep-artifact analyses (from former analyze_results.py)

CLI modes:
  --mode summary          stdout summary tables + compute & val/test gap
  --mode latex            LaTeX tables (main results + compute)
  --mode multi_plots      tradeoff, pareto, latency, speedup, etc. (from JSONs)
  --mode model_plots      t-SNE, denoising trajectory, SVD spectrum (needs ckpts)
  --mode sweep            sensitivity grid + per-config tables
  --mode length_aware     per-bin HR@10 (single run or compare)
  --mode multi_seed_stats statistical comparison of RCCD vs baseline (from sweep dirs)
  --mode pareto_runs      latency Pareto on sweep run names
  --mode outliers         per-user outlier box-plot (test predictions NPZ)
  --mode all              run several main analyses end-to-end

For multi-dataset commands, pass --json_paths a.json b.json c.json ...
For single-dataset commands, pass --dataset <name>.
"""
import argparse
import csv
import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np

# ----- Optional heavy deps loaded lazily where needed -----
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


# ===================================================================
# Section 0: shared style
# ===================================================================

COLORS = {
    'student':    '#1a1a1a',
    'teacher':    '#DAA520',
    'baseline':   '#8B4513',
    'tertiary':   '#B8860B',
    'grid':       '#888888',
    'fill_light': '#F4E4A1',
}
MARKERS = {'student': 'o', 'teacher': 's', 'baseline': '^', 'tertiary': 'D'}


def apply_style():
    if not HAS_PLT:
        return
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
    if not HAS_PLT:
        return
    Path(os.path.dirname(out_path) or '.').mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


# ===================================================================
# Section 1: loaders
# ===================================================================

def load_multiseed_json(path):
    with open(path) as f:
        return json.load(f)


def load_all_sweep_runs(dataset, artifacts_root='artifacts'):
    """Load every per-run summary.json under artifacts/<dataset>/, EXCLUDING
    the special 'teacher/' subdirectory which holds reference.json not summary.json."""
    runs = []
    pattern = os.path.join(artifacts_root, dataset, '*', 'summary.json')
    for path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(path)
        run_name = os.path.basename(run_dir)
        if run_name == 'teacher':
            continue  # safety: should not have summary.json anyway
        with open(path) as f:
            summary = json.load(f)
        config_path = os.path.join(run_dir, 'config.json')
        config = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
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


def load_teacher_config(dataset, artifacts_root='artifacts'):
    """Load teacher's training config (architecture params). Used by model_plots
    to avoid hardcoding hyperparameters."""
    path = os.path.join(artifacts_root, dataset, 'teacher', 'config.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ===================================================================
# Section 2: statistical helpers (from statistics.py)
# ===================================================================

def _gather(results, nfe, metric_key, students_key='students'):
    """Gather metric values from a multi-seed JSON, skipping non-seed keys."""
    out = []
    for seed_key, payload in results[students_key].items():
        if not isinstance(payload, dict):
            continue
        if str(nfe) not in payload:
            continue
        out.append(payload[str(nfe)][metric_key])
    return np.array(out)


def _gather_val(results, nfe, metric_key, students_key='students'):
    out = []
    for seed_key, payload in results[students_key].items():
        if not isinstance(payload, dict) or '_val' not in payload:
            continue
        if str(nfe) not in payload.get('_val', {}):
            continue
        out.append(payload['_val'][str(nfe)][metric_key])
    return np.array(out) if out else None


def gen_gap_analysis(results, nfe, metric='HR@10', students_key='students'):
    student_test = _gather(results, nfe, metric, students_key)
    student_val  = _gather_val(results, nfe, metric, students_key)
    if student_val is None or len(student_val) == 0:
        return None

    teacher_test = results['teacher']['full_nfe'].get(metric)
    teacher_val  = results['teacher'].get('full_nfe_val', {}).get(metric)

    student_gap = student_val.mean() - student_test.mean()
    teacher_gap = (teacher_val - teacher_test) if (teacher_val is not None) else None

    return {
        'student_val_mean':  float(student_val.mean()),
        'student_val_std':   float(student_val.std(ddof=1)) if len(student_val) > 1 else 0.0,
        'student_test_mean': float(student_test.mean()),
        'student_test_std':  float(student_test.std(ddof=1)) if len(student_test) > 1 else 0.0,
        'student_gap':       float(student_gap),
        'teacher_val':       float(teacher_val) if teacher_val is not None else None,
        'teacher_test':      float(teacher_test) if teacher_test is not None else None,
        'teacher_gap':       float(teacher_gap) if teacher_gap is not None else None,
        'gap_ratio':         (float(student_gap / teacher_gap)
                              if (teacher_gap is not None and abs(teacher_gap) > 1e-9)
                              else None),
    }


def gen_gap_analysis_text(results, nfe_grid=(1, 2, 4, 8), metric='HR@10'):
    print(f"\n=== {results['dataset']} | val/test gap | {metric} ===")
    sample = gen_gap_analysis(results, nfe_grid[0], metric)
    if sample is None:
        print('  (no val metrics in JSON)')
        return
    if sample['teacher_val'] is not None:
        print(f"Teacher: val={sample['teacher_val']:.4f}  "
              f"test={sample['teacher_test']:.4f}  "
              f"gap={sample['teacher_gap']:+.4f}")
    header = (f"{'NFE':<5} {'Student val':<22} {'Student test':<22} "
              f"{'Gap':<10} {'Gap/Teacher':<12}")
    print(header); print('-' * len(header))
    for nfe in nfe_grid:
        a = gen_gap_analysis(results, nfe, metric)
        if a is None:
            continue
        gap_ratio = (f"{a['gap_ratio']:.3f}" if a['gap_ratio'] is not None else '--')
        print(f"{nfe:<5} "
              f"{a['student_val_mean']:.4f} ± {a['student_val_std']:.4f}     "
              f"{a['student_test_mean']:.4f} ± {a['student_test_std']:.4f}     "
              f"{a['student_gap']:+.4f}    {gap_ratio}")


def aggregate_seeds(results, nfe_grid=(1, 2, 4, 8),
                    metrics=('HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20'),
                    students_key='students'):
    out = {}
    for nfe in nfe_grid:
        out[nfe] = {}
        for m in metrics:
            vals = _gather(results, nfe, m, students_key)
            if len(vals) == 0:
                out[nfe][m] = {'mean': 0, 'std': 0, 'median': 0,
                               'q25': 0, 'q75': 0, 'values': []}
                continue
            out[nfe][m] = {
                'mean':   float(vals.mean()),
                'std':    float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                'median': float(np.median(vals)),
                'q25':    float(np.percentile(vals, 25)),
                'q75':    float(np.percentile(vals, 75)),
                'values': vals.tolist(),
            }
    return out


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def one_sample_wilcoxon(values, reference):
    """One-sample Wilcoxon: H0 median(values - reference) = 0.
    Returns NaN if n < 5 (Wilcoxon has no power below this)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 5:
        return float('nan')
    residuals = values - reference
    if np.allclose(residuals, 0):
        return 1.0
    try:
        _, p = sst.wilcoxon(residuals)
        return float(p)
    except (ValueError, AttributeError):
        return float('nan')


def paired_wilcoxon(values_a, values_b):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if len(values_a) != len(values_b) or len(values_a) < 5:
        return float('nan')
    if np.allclose(values_a, values_b):
        return 1.0
    try:
        _, p = sst.wilcoxon(values_a, values_b)
        return float(p)
    except (ValueError, AttributeError):
        return float('nan')


def compare_student_vs_baseline(results, nfe, metric='HR@10'):
    student_vals = _gather(results, nfe, metric)
    baseline_val = results['baseline'][str(nfe)][metric]
    n_better = int((student_vals > baseline_val).sum())
    return {
        'student_mean':   float(student_vals.mean()) if len(student_vals) else 0,
        'student_std':    float(student_vals.std(ddof=1)) if len(student_vals) > 1 else 0.0,
        'student_median': float(np.median(student_vals)) if len(student_vals) else 0,
        'baseline':       float(baseline_val),
        'n_seeds':        len(student_vals),
        'n_seeds_better': n_better,
    }


def compare_student_vs_teacher(results, nfe, metric='HR@10'):
    student_vals = _gather(results, nfe, metric)
    teacher_val  = results['teacher']['full_nfe'][metric]
    p = one_sample_wilcoxon(student_vals, teacher_val)
    return {
        'student_mean':       float(student_vals.mean()) if len(student_vals) else 0,
        'student_std':        float(student_vals.std(ddof=1)) if len(student_vals) > 1 else 0.0,
        'student_median':     float(np.median(student_vals)) if len(student_vals) else 0,
        'teacher':            float(teacher_val),
        'gap_pct':            float((student_vals.mean() - teacher_val) / teacher_val * 100)
                              if len(student_vals) else 0,
        'n_seeds':            len(student_vals),
        'n_seeds_above_teacher': int((student_vals > teacher_val).sum()) if len(student_vals) else 0,
        'p_one_sample_wilcoxon': p,
    }


def significance_marker(p):
    """Biostatistics convention: * <0.05, ** <0.01, *** <0.001, **** <0.0001."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    if p < 0.0001: return r'$^{****}$'
    if p < 0.001:  return r'$^{***}$'
    if p < 0.01:   return r'$^{**}$'
    if p < 0.05:   return r'$^{*}$'
    return ''


def compute_metrics(results, nfe_grid=(1, 2, 4, 8, 16, 32)):
    T = results['teacher']['T']
    teacher_full_ms = results['latency']['teacher_full']
    out = {}
    for nfe in nfe_grid:
        nfe_str = str(nfe)
        if nfe_str not in results['latency']['student']:
            continue
        s_lat = results['latency']['student'][nfe_str]
        t_lat = results['latency']['teacher_truncated'][nfe_str]
        out[nfe] = {
            'flops_ratio_vs_teacher_full':  nfe / T,
            'student_throughput_per_sec':   1000.0 / s_lat,
            'teacher_truncated_throughput': 1000.0 / t_lat,
            'student_latency_ms':           s_lat,
            'teacher_truncated_latency_ms': t_lat,
            'speedup_vs_teacher_full':      teacher_full_ms / s_lat,
        }
    out['_teacher_full'] = {
        'latency_ms':         teacher_full_ms,
        'throughput_per_sec': 1000.0 / teacher_full_ms,
        'flops_ratio':        1.0,
        'T':                  T,
    }
    return out


def compute_metrics_text(results, nfe_grid=(1, 2, 4, 8, 16, 32)):
    cm = compute_metrics(results, nfe_grid)
    tf = cm['_teacher_full']
    print(f"\n=== {results['dataset']} | compute & speed ===")
    print(f"Teacher full (NFE={tf['T']}): "
          f"{tf['latency_ms']:.4f} ms, {tf['throughput_per_sec']:.1f} samples/s")
    header = f"{'NFE':<6} {'FLOPs ratio':<14} {'Student ms':<12} {'Throughput':<14} {'Speedup':<10}"
    print(header); print('-' * len(header))
    for nfe in nfe_grid:
        if nfe not in cm:
            continue
        c = cm[nfe]
        print(f"{nfe:<6} {c['flops_ratio_vs_teacher_full']:<14.4f} "
              f"{c['student_latency_ms']:<12.4f} "
              f"{c['student_throughput_per_sec']:<14.1f} "
              f"{c['speedup_vs_teacher_full']:<10.2f}x")


def summary_table_text(results, nfe_grid=(1, 2, 4, 8), metric='HR@10'):
    print(f"\n=== {results['dataset']} | {metric} ===")
    n_seeds = len([k for k in results['students'].keys()
                   if isinstance(results['students'][k], dict)])
    print(f"Teacher (NFE={results['teacher']['T']}): "
          f"{results['teacher']['full_nfe'][metric]:.4f}")
    print(f"(n_seeds = {n_seeds}; Wilcoxon p-values valid only for n >= 5)")
    header = (f"{'NFE':<5} {'Trunc.':<10} {'Student mean±std':<22} "
              f"{'Median (IQR)':<22} {'gap vs T %':<12} "
              f"{'p_vs_teacher':<14} {'p_vs_trunc':<12}")
    print(header); print('-' * len(header))
    for nfe in nfe_grid:
        student_vals = _gather(results, nfe, metric)
        if len(student_vals) == 0:
            continue
        teacher_val  = results['teacher']['full_nfe'][metric]
        truncated_val = results['baseline'][str(nfe)][metric]
        p_vs_teacher = one_sample_wilcoxon(student_vals, teacher_val)
        p_vs_trunc   = one_sample_wilcoxon(student_vals, truncated_val)
        gap_pct = (student_vals.mean() - teacher_val) / teacher_val * 100
        mean = student_vals.mean()
        std  = student_vals.std(ddof=1) if len(student_vals) > 1 else 0
        med  = np.median(student_vals)
        q25  = np.percentile(student_vals, 25)
        q75  = np.percentile(student_vals, 75)
        print(f"{nfe:<5} {truncated_val:<10.4f} "
              f"{mean:.4f} ± {std:.4f}    "
              f"{med:.4f} ({q25:.4f}-{q75:.4f})    "
              f"{gap_pct:>+8.2f}%   "
              f"{p_vs_teacher:<14.4g} {p_vs_trunc:<12.4g}")


# ===================================================================
# Section 3: LaTeX tables
# ===================================================================

def latex_main_table(results_per_dataset, nfe=1,
                     metrics=('HR@5', 'HR@10', 'HR@20',
                              'NDCG@5', 'NDCG@10', 'NDCG@20')):
    try:
        from literature_baselines import LITERATURE_BASELINES, BASELINE_ORDER, DISPLAY_NAMES
    except ImportError:
        LITERATURE_BASELINES, BASELINE_ORDER, DISPLAY_NAMES = {}, [], {}

    lines = []
    lines.append(r'\begin{tabular}{l l ' + 'r' * len(metrics) + '}')
    lines.append(r'\toprule')
    lines.append('Dataset & Method & ' + ' & '.join(metrics) + r' \\')
    lines.append(r'\midrule')

    for ds_name, res in results_per_dataset.items():
        agg = aggregate_seeds(res, nfe_grid=[nfe], metrics=metrics)
        n_seeds = len([k for k in res['students'].keys()
                       if isinstance(res['students'][k], dict)])

        lit = LITERATURE_BASELINES.get(ds_name, {})
        if lit:
            for i, baseline_name in enumerate(BASELINE_ORDER):
                if baseline_name not in lit:
                    continue
                pretty = DISPLAY_NAMES.get(baseline_name, baseline_name)
                row_cells = ['', pretty]
                for m in metrics:
                    val = lit[baseline_name].get(m)
                    row_cells.append(f'{val:.2f}' if val is not None else '--')
                prefix = ds_name if i == 0 else ''
                lines.append(prefix + ' & ' + ' & '.join(row_cells[1:]) + r' \\')
            lines.append(r'\cmidrule(lr){2-' + str(len(metrics) + 2) + '}')
            ds_prefix = ''
        else:
            ds_prefix = ds_name

        teacher_row = ['', f'Teacher (NFE={res["teacher"]["T"]}, ours)']
        for m in metrics:
            teacher_row.append(f'{res["teacher"]["full_nfe"][m]:.2f}')
        lines.append(ds_prefix + ' & ' + ' & '.join(teacher_row[1:]) + r' \\')

        baseline_row = ['', f'Truncated DDIM (NFE={nfe})']
        for m in metrics:
            baseline_row.append(f'{res["baseline"][str(nfe)][m]:.2f}')
        lines.append(' & ' + ' & '.join(baseline_row[1:]) + r' \\')

        # CD baseline if present in this multi-seed JSON
        if 'students_baseline' in res and res['students_baseline']:
            agg_base = aggregate_seeds(res, nfe_grid=[nfe], metrics=metrics,
                                       students_key='students_baseline')
            row = ['', f'CD baseline (NFE={nfe}, mean$\\pm$std)']
            for m in metrics:
                row.append(f'{agg_base[nfe][m]["mean"]:.2f} $\\pm$ {agg_base[nfe][m]["std"]:.2f}')
            lines.append(' & ' + ' & '.join(row[1:]) + r' \\')

        student_row = ['', f'RCCD student (NFE={nfe}, mean$\\pm$std)']
        for m in metrics:
            mean = agg[nfe][m]['mean']
            std  = agg[nfe][m]['std']
            vals = np.array(agg[nfe][m]['values'])
            if len(vals) == 0:
                student_row.append('--')
                continue
            base_val = res['baseline'][str(nfe)][m]
            p_trunc   = one_sample_wilcoxon(vals, base_val)
            p_teacher = one_sample_wilcoxon(vals, res['teacher']['full_nfe'][m])
            star = significance_marker(p_trunc)
            dagger = r'$^{\dagger}$' if (not np.isnan(p_teacher) and p_teacher >= 0.05) else ''
            student_row.append(f'{mean:.2f}{star}{dagger} $\\pm$ {std:.2f}')
        lines.append(' & ' + ' & '.join(student_row[1:]) + r' \\')
        lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append('')
    lines.append(r'% Significance markers:')
    lines.append(r'%   *,**,***,**** — student significantly beats Truncated DDIM (Wilcoxon p<0.05/0.01/0.001/0.0001).')
    lines.append(r'%   $\dagger$ — no significant difference from teacher full NFE (one-sample Wilcoxon p>=0.05); desirable.')
    lines.append(r'%   With n=3 seeds Wilcoxon is underpowered; markers may be absent even when descriptive trend is clear.')
    return '\n'.join(lines)


def latex_compute_table(results_per_dataset, nfe_grid=(1, 2, 4, 8)):
    lines = []
    lines.append(r'\begin{tabular}{l l r r r r}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Method & Latency (ms) & Throughput (samples/s) '
                 r'& FLOPs ratio & Speedup \\')
    lines.append(r'\midrule')
    for ds_name, res in results_per_dataset.items():
        cm = compute_metrics(res, nfe_grid)
        tf = cm['_teacher_full']
        lines.append(f'{ds_name} & Teacher (NFE={tf["T"]}) '
                     f'& {tf["latency_ms"]:.3f} '
                     f'& {tf["throughput_per_sec"]:.1f} '
                     f'& {tf["flops_ratio"]:.3f} '
                     f'& 1.00$\\times$ \\\\')
        for nfe in nfe_grid:
            if nfe not in cm:
                continue
            c = cm[nfe]
            lines.append(f' & Student (NFE={nfe}) '
                         f'& {c["student_latency_ms"]:.3f} '
                         f'& {c["student_throughput_per_sec"]:.1f} '
                         f'& {c["flops_ratio_vs_teacher_full"]:.3f} '
                         f'& {c["speedup_vs_teacher_full"]:.2f}$\\times$ \\\\')
        lines.append(r'\midrule')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)


# ===================================================================
# Section 4: multi-dataset plots (from plots.py)
# ===================================================================

def _student_mean_std(results, nfe_grid, metric):
    means, stds = [], []
    for nfe in nfe_grid:
        vals = _gather(results, nfe, metric)
        if len(vals) == 0:
            means.append(np.nan); stds.append(0); continue
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(stds)


def plot_tradeoff_nfe_quality(results_per_dataset, metric='HR@10',
                              out_path='figures/tradeoff_nfe_quality.pdf'):
    if not HAS_PLT: return
    apply_style()
    n = len(results_per_dataset)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (ds_name, res) in zip(axes, results_per_dataset.items()):
        nfe_grid = sorted(int(k) for k in res['baseline'].keys())
        ax.axhline(res['teacher']['full_nfe'][metric], color=COLORS['teacher'],
                   linestyle=':', linewidth=1.8,
                   label=f"Teacher (NFE={res['teacher']['T']})")
        truncated = [res['baseline'][str(n_)][metric] for n_ in nfe_grid]
        ax.plot(nfe_grid, truncated, marker=MARKERS['baseline'],
                color=COLORS['baseline'], linewidth=2.0, label='Truncated DDIM')
        means, stds = _student_mean_std(res, nfe_grid, metric)
        ax.errorbar(nfe_grid, means, yerr=stds, marker=MARKERS['student'],
                    color=COLORS['student'], linewidth=2.0, capsize=4,
                    label='RCCD student')
        ax.fill_between(nfe_grid, means - stds, means + stds,
                        color=COLORS['fill_light'], alpha=0.4)
        ax.set_xscale('log', base=2)
        ax.set_xticks(nfe_grid)
        ax.set_xticklabels(nfe_grid)
        ax.set_xlabel('NFE'); ax.set_ylabel(metric)
        ax.set_title(ds_name)
        ax.legend(loc='lower right', framealpha=0.95)
    fig.suptitle(f'Quality vs inference cost ({metric})', y=1.02, fontsize=13)
    _save(fig, out_path)


def _pareto_mask(latencies, qualities, maximize_quality=True):
    """Return boolean mask: True for points on the Pareto frontier
    (lower latency is better, higher quality is better)."""
    lat = np.array(latencies); q = np.array(qualities)
    n = len(lat)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            better_lat = lat[j] <= lat[i]
            better_q   = q[j] >= q[i] if maximize_quality else q[j] <= q[i]
            strictly   = (lat[j] < lat[i]) or (q[j] > q[i] if maximize_quality else q[j] < q[i])
            if better_lat and better_q and strictly:
                is_pareto[i] = False
                break
    return is_pareto


def plot_pareto(results, metric='HR@10', out_path='figures/pareto.pdf'):
    if not HAS_PLT: return
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    nfe_grid = sorted(int(k) for k in results['latency']['student'].keys())

    s_means, s_stds = _student_mean_std(results, nfe_grid, metric)
    s_lat = [results['latency']['student'][str(n_)] for n_ in nfe_grid]

    pareto_mask = _pareto_mask(s_lat, s_means.tolist())

    ax.errorbar(s_lat, s_means, yerr=s_stds, marker=MARKERS['student'],
                color=COLORS['student'], linewidth=2.0, capsize=4,
                label='RCCD student')
    ax.scatter(np.array(s_lat)[pareto_mask], s_means[pareto_mask],
               s=180, facecolors='none', edgecolors='red', linewidths=2.0,
               zorder=6, label='Pareto-optimal')
    for nfe, x_, y_ in zip(nfe_grid, s_lat, s_means):
        ax.annotate(f'NFE={nfe}', (x_, y_), textcoords='offset points',
                    xytext=(6, 6), fontsize=8, color=COLORS['student'])

    b_q   = [results['baseline'][str(n_)][metric] for n_ in nfe_grid]
    b_lat = [results['latency']['teacher_truncated'][str(n_)] for n_ in nfe_grid]
    ax.plot(b_lat, b_q, marker=MARKERS['baseline'], color=COLORS['baseline'],
            linewidth=2.0, label='Truncated DDIM')

    ax.scatter([results['latency']['teacher_full']],
               [results['teacher']['full_nfe'][metric]],
               marker='*', s=200, color=COLORS['teacher'],
               label=f"Teacher full (NFE={results['teacher']['T']})",
               zorder=5, edgecolors='black', linewidths=0.8)

    ax.set_xscale('log')
    ax.set_xlabel('Inference latency (ms / sample)')
    ax.set_ylabel(metric)
    ax.set_title(f'Pareto: quality vs latency — {results["dataset"]}')
    ax.legend(loc='lower right', fontsize=9)
    _save(fig, out_path)


def plot_latency_bars(results, out_path='figures/latency_bars.pdf'):
    if not HAS_PLT: return
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    nfe_grid = sorted(int(k) for k in results['latency']['student'].keys())
    x = np.arange(len(nfe_grid)); w = 0.4
    s_lat = [results['latency']['student'][str(n_)] for n_ in nfe_grid]
    t_lat = [results['latency']['teacher_truncated'][str(n_)] for n_ in nfe_grid]
    ax.bar(x - w/2, t_lat, w, color=COLORS['baseline'], label='Truncated DDIM')
    ax.bar(x + w/2, s_lat, w, color=COLORS['student'], label='RCCD student')
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


def plot_speedup(results_per_dataset, out_path='figures/speedup.pdf'):
    if not HAS_PLT: return
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    datasets = list(results_per_dataset.keys())
    first = next(iter(results_per_dataset.values()))
    nfe_grid = sorted(int(k) for k in first['latency']['student'].keys())
    x = np.arange(len(datasets))
    w = 0.8 / max(len(nfe_grid), 1)
    cmap = [COLORS['student'], COLORS['tertiary'], COLORS['baseline'], COLORS['teacher']]
    for i, nfe in enumerate(nfe_grid[:4]):
        speedups = []
        for ds, res in results_per_dataset.items():
            speedups.append(res['latency']['teacher_full'] /
                            res['latency']['student'][str(nfe)])
        ax.bar(x + (i - len(nfe_grid[:4])/2 + 0.5) * w, speedups, w,
               label=f'NFE={nfe}', color=cmap[i % len(cmap)])
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel(r'Speedup vs teacher full ($\times$)')
    ax.set_title('Inference speedup of distilled student')
    ax.legend()
    _save(fig, out_path)


def plot_main_results_errorbars(results_per_dataset, metric='HR@10',
                                out_path='figures/main_results_errorbars.pdf'):
    if not HAS_PLT: return
    apply_style()
    datasets = list(results_per_dataset.keys())
    n = len(datasets)
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * n), 4.5))
    x = np.arange(n); w = 0.27
    teacher_vals  = [r['teacher']['full_nfe'][metric] for r in results_per_dataset.values()]
    baseline_vals = [r['baseline']['1'][metric] for r in results_per_dataset.values()]
    student_means, student_stds = [], []
    for r in results_per_dataset.values():
        v = _gather(r, 1, metric)
        student_means.append(v.mean() if len(v) else 0)
        student_stds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
    ax.bar(x - w, teacher_vals, w,  color=COLORS['teacher'], label='Teacher (NFE=T)',
           edgecolor='black', linewidth=0.5)
    ax.bar(x,     baseline_vals, w, color=COLORS['baseline'], label='Truncated DDIM (NFE=1)',
           edgecolor='black', linewidth=0.5)
    ax.bar(x + w, student_means, w, yerr=student_stds, capsize=5,
           color=COLORS['student'], label='RCCD student (NFE=1)',
           edgecolor='black', linewidth=0.5,
           error_kw={'ecolor': COLORS['tertiary'], 'elinewidth': 1.5})
    ax.set_xticks(x); ax.set_xticklabels(datasets); ax.set_ylabel(metric)
    ax.set_title(f'Main results ({metric}, mean $\\pm$ std)')
    ax.legend()
    _save(fig, out_path)


def plot_bootstrap_ci(results, metric='HR@10', n_boot=10000,
                      out_path='figures/bootstrap_ci.pdf'):
    if not HAS_PLT: return
    apply_style()
    nfe_grid = sorted(int(k) for k in results['baseline'].keys())
    means, los, his, base_vals = [], [], [], []
    rng = np.random.default_rng(0)
    for nfe in nfe_grid:
        v = _gather(results, nfe, metric)
        if len(v) == 0:
            means.append(np.nan); los.append(np.nan); his.append(np.nan)
        else:
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
    ax.set_xticks(nfe_grid); ax.set_xticklabels(nfe_grid)
    ax.set_xlabel('NFE'); ax.set_ylabel(metric)
    ax.set_title(f'95% bootstrap CI — {results["dataset"]}')
    ax.legend()
    _save(fig, out_path)


def plot_val_vs_test_gap(results, metric='HR@10',
                         nfe_grid=(1, 2, 4, 8, 16, 32),
                         out_path='figures/val_test_gap.pdf'):
    if not HAS_PLT: return
    apply_style()
    val_means, val_stds, test_means, test_stds = [], [], [], []
    nfe_used = []
    for nfe in nfe_grid:
        test_v = _gather(results, nfe, metric)
        val_v = _gather_val(results, nfe, metric)
        if val_v is None or len(val_v) == 0 or len(test_v) == 0:
            continue
        nfe_used.append(nfe)
        val_means.append(val_v.mean())
        val_stds.append(val_v.std(ddof=1) if len(val_v) > 1 else 0.0)
        test_means.append(test_v.mean())
        test_stds.append(test_v.std(ddof=1) if len(test_v) > 1 else 0.0)
    if not nfe_used:
        return
    val_means = np.array(val_means); val_stds = np.array(val_stds)
    test_means = np.array(test_means); test_stds = np.array(test_stds)
    x = np.arange(len(nfe_used)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w / 2, val_means, w, yerr=val_stds, capsize=3,
           color=COLORS['student'], alpha=0.85,
           edgecolor='black', linewidth=0.5, label='Student — Val')
    ax.bar(x + w / 2, test_means, w, yerr=test_stds, capsize=3,
           color=COLORS['fill_light'], alpha=0.95,
           edgecolor='black', linewidth=0.5, label='Student — Test')
    teacher_test = results['teacher']['full_nfe'].get(metric)
    teacher_val  = results['teacher'].get('full_nfe_val', {}).get(metric)
    if teacher_test is not None:
        ax.axhline(teacher_test, color=COLORS['teacher'],
                   linestyle=':', linewidth=1.8,
                   label=f'Teacher — Test ({teacher_test:.2f})')
    if teacher_val is not None:
        ax.axhline(teacher_val, color=COLORS['baseline'],
                   linestyle='--', linewidth=1.5, alpha=0.8,
                   label=f'Teacher — Val ({teacher_val:.2f})')
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in nfe_used])
    ax.set_xlabel('NFE'); ax.set_ylabel(metric)
    ax.set_title(f'Val vs test {metric} — {results["dataset"]}')
    ax.legend(loc='best', fontsize=9)
    _save(fig, out_path)


def plot_convergence(log_dir, out_path='figures/convergence.pdf', skip_warmup=0):
    if not HAS_PLT: return
    csv_paths = sorted(glob.glob(os.path.join(log_dir, '*.csv')))
    csv_paths = [p for p in csv_paths if not p.endswith('.val.csv')]
    if not csv_paths:
        print(f'[plot_convergence] no CSVs in {log_dir}')
        return
    series = []
    for p in csv_paths:
        ep, cons, ce, total = [], [], [], []
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
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
    cons_mat = _stack(1); ce_mat = _stack(2); total_mat = _stack(3)
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
        ax.plot(e, m, color=color, linewidth=2.0, label='mean across runs')
        ax.fill_between(e, m - s, m + s, color=COLORS['fill_light'], alpha=0.5,
                        label=r'$\pm$ 1 std')
        ax.set_xlabel('Epoch'); ax.set_ylabel(title); ax.set_title(title)
        ax.legend(loc='upper right', fontsize=9)
    fig.suptitle(f'Training convergence ({len(series)} runs)', y=1.02)
    _save(fig, out_path)


# ===================================================================
# Section 5: model-aware plots (from analysis_plots.py)
# ===================================================================

def _build_args_from_teacher_config(dataset, data_root, artifacts_root='artifacts'):
    """Reconstruct args from saved teacher config.json.
    Falls back to defaults if config is missing."""
    import torch
    class _A: pass
    a = _A()
    cfg = load_teacher_config(dataset, artifacts_root) or {}
    a.dataset = dataset
    a.data_root = data_root
    a.max_len = int(cfg.get('max_len', 50))
    a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a.batch_size = int(cfg.get('batch_size', 256))
    a.hidden_size = int(cfg.get('hidden_size', 128))
    a.dropout = float(cfg.get('dropout', 0.1))
    a.emb_dropout = float(cfg.get('emb_dropout', 0.3))
    a.hidden_act = str(cfg.get('hidden_act', 'gelu'))
    a.num_blocks = int(cfg.get('num_blocks', 4))
    a.diffusion_steps = int(cfg.get('diffusion_steps', 32))
    a.lambda_uncertainty = float(cfg.get('lambda_uncertainty', 0.001))
    a.noise_schedule = str(cfg.get('noise_schedule', 'trunc_lin'))
    a.rescale_timesteps = cfg.get('rescale_timesteps', True)
    a.schedule_sampler_name = str(cfg.get('schedule_sampler_name', 'lossaware'))
    return a


def _load_models_and_data(args_ns, teacher_ckpt, student_ckpt):
    import torch
    from utils import Data_Test
    from model import create_model_diffu, Att_Diffuse_model
    from consistency_diffurec import ConsistencyStudent

    path = os.path.join(args_ns.data_root, args_ns.dataset, 'dataset.pkl')
    with open(path, 'rb') as f:
        data_raw = pickle.load(f)
    args_ns.item_num = len(data_raw['smap'])

    device = torch.device(args_ns.device)
    teacher = Att_Diffuse_model(create_model_diffu(args_ns), args_ns).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = ConsistencyStudent(teacher, args_ns).to(device)
    student.load_state_dict(torch.load(student_ckpt, map_location=device))
    student.eval()

    test_loader = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'],
                            args_ns).get_pytorch_dataloaders()
    return teacher, student, test_loader, data_raw


def plot_tsne_embeddings(teacher, student, n_items=2000, n_clusters=8,
                         out_path='figures/tsne_embeddings.pdf', seed=0):
    if not HAS_PLT: return
    import torch
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    apply_style()
    np.random.seed(seed)
    with torch.no_grad():
        e_teacher = teacher.item_embeddings.weight.detach().cpu().numpy()
        e_student = student.item_embeddings.weight.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(min(e_teacher.shape[0], e_student.shape[0]),
                     size=min(n_items, e_teacher.shape[0]), replace=False)
    et = e_teacher[idx]; es = e_student[idx]
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(et)
    print('[t-SNE] computing teacher projection...')
    proj_t = TSNE(n_components=2, random_state=seed, init='pca',
                  perplexity=30, learning_rate='auto').fit_transform(et)
    print('[t-SNE] computing student projection...')
    proj_s = TSNE(n_components=2, random_state=seed, init='pca',
                  perplexity=30, learning_rate='auto').fit_transform(es)
    cmap = plt.get_cmap('cividis')(np.linspace(0.05, 0.95, n_clusters))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, proj, title in [(axes[0], proj_t, 'Teacher item embeddings'),
                            (axes[1], proj_s, 'Student item embeddings')]:
        for c in range(n_clusters):
            m = labels == c
            ax.scatter(proj[m, 0], proj[m, 1], s=8, alpha=0.7,
                       color=cmap[c], edgecolor='none')
        ax.set_title(title); ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Embedding structure preservation under consistency distillation', y=1.02)
    _save(fig, out_path)


def plot_denoising_trajectory(teacher, student, test_loader, device,
                              sample_idx=0,
                              out_path='figures/denoising_trajectory.pdf', seed=0):
    if not HAS_PLT: return
    import torch
    from sklearn.decomposition import PCA
    apply_style()
    torch.manual_seed(seed); np.random.seed(seed)
    sample_seq, _ = next(iter(test_loader))
    if sample_idx >= sample_seq.size(0):
        print(f'[trajectory] sample_idx out of range, using 0'); sample_idx = 0
    sample_seq = sample_seq[sample_idx:sample_idx + 1].to(device)
    target_idx = sample_seq[0, -1].item()
    if target_idx == 0:
        nz = sample_seq[0].nonzero().flatten()
        target_idx = sample_seq[0, nz[-1]].item()
    e_target = teacher.item_embeddings.weight[target_idx].detach().cpu().numpy()

    diffu = teacher.diffu
    item_emb = teacher.item_embeddings(sample_seq)
    item_emb = teacher.embed_dropout(item_emb)
    item_emb = teacher.LayerNorm(item_emb)
    mask_seq = (sample_seq > 0).float()
    H = item_emb.size(-1); T = diffu.num_timesteps
    x_t = torch.randn(1, H, device=device)
    trajectory = []
    indices = list(range(T))[::-1]
    with torch.no_grad():
        for i in indices:
            t = torch.tensor([i], device=device, dtype=torch.long)
            x_0_pred, _ = diffu.xstart_model(item_emb, x_t, diffu._scale_timesteps(t), mask_seq)
            trajectory.append(x_0_pred[0].cpu().numpy())
            x_t = diffu.p_sample(item_emb, x_t, t, mask_seq)
    traj_teacher = np.stack(trajectory)

    item_rep, mask_seq = student.encode(sample_seq)
    x_t = torch.randn(1, H, device=device)
    t = torch.full((1,), T - 1, device=device, dtype=torch.long)
    with torch.no_grad():
        x_0 = student.diffu_student.predict_x0(item_rep, x_t, t, mask_seq)
    pred_student = x_0[0].cpu().numpy()

    all_pts = np.vstack([traj_teacher, pred_student[None, :], e_target[None, :]])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_pts)
    proj_teacher = proj[:len(traj_teacher)]
    proj_student = proj[len(traj_teacher)]
    proj_target  = proj[-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(proj_teacher[:, 0], proj_teacher[:, 1],
            color=COLORS['teacher'], linewidth=1.5, alpha=0.7,
            label='Teacher trajectory (32 steps)')
    n_pts = len(proj_teacher)
    sc = ax.scatter(proj_teacher[:, 0], proj_teacher[:, 1],
                    c=np.arange(n_pts), cmap='cividis', s=30,
                    edgecolor='black', linewidth=0.4, zorder=3)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label('Reverse step (T-1 → 0)')
    ax.scatter(proj_teacher[0, 0], proj_teacher[0, 1],
               marker='X', s=180, color=COLORS['baseline'],
               edgecolor='black', linewidth=0.8, zorder=4,
               label='Teacher start (NFE=T)')
    ax.scatter(proj_teacher[-1, 0], proj_teacher[-1, 1],
               marker='*', s=240, color=COLORS['teacher'],
               edgecolor='black', linewidth=0.8, zorder=4,
               label='Teacher end (x₀)')
    ax.scatter(proj_student[0], proj_student[1],
               marker='o', s=200, color=COLORS['student'],
               edgecolor='black', linewidth=0.8, zorder=5,
               label='Student (NFE=1)')
    ax.scatter(proj_target[0], proj_target[1],
               marker='D', s=140, color=COLORS['tertiary'],
               edgecolor='black', linewidth=0.8, zorder=4,
               label='True target embedding')
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
    ax.set_title(f'Denoising trajectory (fixed sample idx={sample_idx})')
    ax.legend(loc='best', fontsize=9)
    _save(fig, out_path)


def plot_svd_spectrum(teacher, student, out_path='figures/svd_spectrum.pdf', top_k=10):
    if not HAS_PLT: return
    import torch
    from scipy.linalg import subspace_angles
    apply_style()
    with torch.no_grad():
        et = teacher.item_embeddings.weight.detach().cpu().numpy()
        es = student.item_embeddings.weight.detach().cpu().numpy()
    Ut, s_t, _ = np.linalg.svd(et, full_matrices=False)
    Us, s_s, _ = np.linalg.svd(es, full_matrices=False)
    k = min(top_k, Ut.shape[1], Us.shape[1])
    angles_rad = subspace_angles(Ut[:, :k], Us[:, :k])
    angles_deg = np.rad2deg(angles_rad)
    max_angle = float(angles_deg.max())
    mean_angle = float(angles_deg.mean())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    ax.plot(np.arange(len(s_t)), s_t, color=COLORS['teacher'], linewidth=2.0, label='Teacher')
    ax.plot(np.arange(len(s_s)), s_s, color=COLORS['student'], linewidth=2.0,
            linestyle='--', label='Student')
    ax.set_xlabel('Singular value index'); ax.set_ylabel('Singular value (linear)')
    ax.set_title('Spectrum (linear)'); ax.legend()

    ax = axes[1]
    ax.semilogy(np.arange(len(s_t)), s_t, color=COLORS['teacher'], linewidth=2.0, label='Teacher')
    ax.semilogy(np.arange(len(s_s)), s_s, color=COLORS['student'], linewidth=2.0,
                linestyle='--', label='Student')
    ax.set_xlabel('Singular value index'); ax.set_ylabel('Singular value (log)')
    ax.set_title('Spectrum (log)'); ax.legend()

    ax = axes[2]
    ax.bar(np.arange(k), angles_deg, color=COLORS['student'],
           edgecolor='black', linewidth=0.5)
    ax.axhline(y=90, color='red', linestyle=':', alpha=0.5, label='90° (orthogonal)')
    ax.set_xlabel('Principal direction index'); ax.set_ylabel('Subspace angle (°)')
    ax.set_title(f'Top-{k} subspace angles\n(0° = aligned, 90° = orthogonal)')
    ax.set_ylim(0, 95); ax.legend(fontsize=8)

    cos_sim_spectrum = np.dot(s_t, s_s) / (np.linalg.norm(s_t) * np.linalg.norm(s_s))
    rel_diff_spectrum = np.linalg.norm(s_t - s_s) / np.linalg.norm(s_t)
    fig.suptitle(f'Embedding spectrum: cos sim = {cos_sim_spectrum:.4f}, '
                 f'rel L2 diff = {rel_diff_spectrum:.4f}. '
                 f'Top-{k} subspace angles: max = {max_angle:.1f}°, mean = {mean_angle:.1f}°',
                 y=1.02, fontsize=10)
    _save(fig, out_path)


# ===================================================================
# Section 6: sweep analyses (from analyze_results.py)
# ===================================================================

def sensitivity_analysis(dataset, metric='HR@10', nfe='1',
                         artifacts_root='artifacts',
                         out_dir='figures', seed_filter=None):
    runs = load_all_sweep_runs(dataset, artifacts_root)
    if seed_filter is not None:
        runs = [r for r in runs if r['random_seed'] == seed_filter]
    grid = {}
    for r in runs:
        beta = r['contrast_weight']; tau = r['contrast_temperature']
        val = r['test_metrics_per_nfe'][nfe][metric]
        grid[(beta, tau)] = val
    betas = sorted({b for b, _ in grid})
    taus = sorted({t for _, t in grid})
    matrix = np.full((len(betas), len(taus)), np.nan)
    for i, b in enumerate(betas):
        for j, t in enumerate(taus):
            if (b, t) in grid:
                matrix[i, j] = grid[(b, t)]
    print(f'\n=== Sensitivity grid for {dataset}, {metric}@NFE={nfe} ===')
    header = '          ' + '  '.join(f'tau={t:<6}' for t in taus)
    print(header)
    for i, b in enumerate(betas):
        row = f'beta={b:<5} '
        for j in range(len(taus)):
            v = matrix[i, j]
            row += f'{v:>9.4f}  ' if not np.isnan(v) else '    --     '
        print(row)
    if not np.all(np.isnan(matrix)):
        best_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
        best_beta = betas[best_idx[0]]; best_tau = taus[best_idx[1]]
        best_val = matrix[best_idx]
        print(f'\nBest: beta={best_beta}, tau={best_tau} -> {metric}={best_val:.4f}')
    if HAS_PLT:
        apply_style()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(matrix, aspect='auto', cmap='cividis')
        ax.set_xticks(range(len(taus))); ax.set_xticklabels([f'{t}' for t in taus])
        ax.set_yticks(range(len(betas))); ax.set_yticklabels([f'{b}' for b in betas])
        ax.set_xlabel('Contrast temperature τ')
        ax.set_ylabel('Contrast weight β')
        ax.set_title(f'{metric} @ NFE={nfe} ({dataset})')
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


def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z*z/n
    center = (p + z*z/(2*n)) / denom
    spread = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, center - spread), min(1, center + spread))


def length_aware_analysis(dataset, run_name, artifacts_root='artifacts',
                          out_dir='figures', n_bins=5):
    run_dir = os.path.join(artifacts_root, dataset, run_name)
    pred_path = os.path.join(run_dir, 'test_predictions_nfe1.npz')
    if not os.path.exists(pred_path):
        print(f'[length-aware] no predictions at {pred_path}')
        return None
    data = np.load(pred_path)
    lengths = data['hist_lengths']; ks = data['ks']
    hits = data['hit_at_k']
    try:
        k10_idx = list(ks).index(10)
    except ValueError:
        k10_idx = 1
    quantiles = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    bin_results = []
    for i in range(n_bins):
        lo, hi = quantiles[i], quantiles[i + 1]
        if i == n_bins - 1:
            mask = (lengths >= lo) & (lengths <= hi)
        else:
            mask = (lengths >= lo) & (lengths < hi)
        n = int(mask.sum())
        if n == 0: continue
        hit_rate = float(hits[mask, k10_idx].mean()) * 100
        k_hits = int(hits[mask, k10_idx].sum())
        ci_lo, ci_hi = wilson_ci(k_hits, n)
        ci_lo, ci_hi = ci_lo * 100, ci_hi * 100
        bin_results.append({
            'bin': i+1, 'len_lo': int(lo), 'len_hi': int(hi),
            'n': n, 'HR@10': hit_rate, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        })
    print(f'\n=== Length-aware analysis: {dataset} / {run_name} ===')
    print(f'{"Bin":<5} {"Range":<15} {"n":<6} {"HR@10":<8} {"95% Wilson CI":<20}')
    for r in bin_results:
        print(f'{r["bin"]:<5} {r["len_lo"]}-{r["len_hi"]:<10} {r["n"]:<6} '
              f'{r["HR@10"]:<8.2f} [{r["ci_lo"]:.2f}, {r["ci_hi"]:.2f}]')
    if HAS_PLT and bin_results:
        apply_style()
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [f'{r["len_lo"]}-{r["len_hi"]}' for r in bin_results]
        ys = [r['HR@10'] for r in bin_results]
        ylo = [r['HR@10'] - r['ci_lo'] for r in bin_results]
        yhi = [r['ci_hi'] - r['HR@10'] for r in bin_results]
        ax.bar(xs, ys, yerr=[ylo, yhi], capsize=4,
               color=COLORS['student'], edgecolor='black', linewidth=0.5)
        ax.set_xlabel('History length range'); ax.set_ylabel('HR@10 (%)')
        ax.set_title(f'Performance by length — {run_name}')
        _save(fig, os.path.join(out_dir, f'length_aware_{dataset}_{run_name}.pdf'))
    return bin_results


def length_aware_compare(dataset, run_names, artifacts_root='artifacts',
                         out_dir='figures', n_bins=5, labels=None):
    all_results = {}
    for rn in run_names:
        res = length_aware_analysis(dataset, rn, artifacts_root,
                                    out_dir=out_dir, n_bins=n_bins)
        if res is not None:
            all_results[rn] = res
    if not HAS_PLT or len(all_results) < 2:
        return all_results
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    runs_list = list(all_results.keys())
    bins_x = [f'{r["len_lo"]}-{r["len_hi"]}' for r in all_results[runs_list[0]]]
    width = 0.8 / len(runs_list); x = np.arange(len(bins_x))
    palette = [COLORS['baseline'], COLORS['student'], COLORS['teacher']]
    for i, rn in enumerate(runs_list):
        ys = [r['HR@10'] for r in all_results[rn]]
        label = labels[i] if labels else rn
        ax.bar(x + (i - len(runs_list)/2 + 0.5)*width, ys, width,
               color=palette[i % len(palette)], edgecolor='black',
               linewidth=0.5, label=label)
    ax.set_xticks(x); ax.set_xticklabels(bins_x)
    ax.set_xlabel('History length range'); ax.set_ylabel('HR@10 (%)')
    ax.set_title(f'Length-aware comparison ({dataset})'); ax.legend()
    _save(fig, os.path.join(out_dir, f'length_aware_compare_{dataset}.pdf'))
    return all_results


def multi_seed_statistics_from_sweep(dataset, beta_target, tau_target,
                                     beta_baseline=0.0, tau_baseline=0.1,
                                     metric='HR@10', nfe='1',
                                     artifacts_root='artifacts', n_boot=10000):
    runs = load_all_sweep_runs(dataset, artifacts_root)
    rccd_vals = [r['test_metrics_per_nfe'][nfe][metric]
                 for r in runs
                 if abs(r['contrast_weight'] - beta_target) < 1e-9
                 and abs(r['contrast_temperature'] - tau_target) < 1e-9]
    base_vals = [r['test_metrics_per_nfe'][nfe][metric]
                 for r in runs
                 if abs(r['contrast_weight'] - beta_baseline) < 1e-9
                 and abs(r['contrast_temperature'] - tau_baseline) < 1e-9]
    print(f'\n=== Multi-seed stats from sweep dirs: {dataset}, {metric}@NFE={nfe} ===')
    print(f'RCCD  (β={beta_target}, τ={tau_target}): n={len(rccd_vals)}, vals={rccd_vals}')
    print(f'Base  (β=0):                             n={len(base_vals)}, vals={base_vals}')
    if len(rccd_vals) == 0 or len(base_vals) == 0:
        print('Not enough runs.'); return None
    rccd_arr = np.array(rccd_vals); base_arr = np.array(base_vals)
    rccd_mean = rccd_arr.mean(); rccd_std = rccd_arr.std(ddof=1) if len(rccd_arr) > 1 else 0
    base_mean = base_arr.mean(); base_std = base_arr.std(ddof=1) if len(base_arr) > 1 else 0
    diff = rccd_mean - base_mean
    print(f'\nRCCD:     {rccd_mean:.4f} ± {rccd_std:.4f}')
    print(f'Baseline: {base_mean:.4f} ± {base_std:.4f}')
    print(f'Diff:     {diff:+.4f}')
    if HAS_SCIPY and len(rccd_arr) == len(base_arr) and len(rccd_arr) >= 5:
        try:
            stat, p = sst.wilcoxon(rccd_arr, base_arr)
            print(f'Paired Wilcoxon: stat={stat:.3f}, p={p:.4f}')
        except ValueError as e:
            print(f'Wilcoxon error: {e}')
    if len(rccd_arr) >= 2 and len(base_arr) >= 2:
        rng = np.random.default_rng(0)
        diffs = np.array([
            rng.choice(rccd_arr, len(rccd_arr), replace=True).mean() -
            rng.choice(base_arr, len(base_arr), replace=True).mean()
            for _ in range(n_boot)
        ])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f'Bootstrap 95% CI on diff: [{lo:+.4f}, {hi:+.4f}]')
        print(f'P(RCCD > baseline) = {(diffs > 0).mean():.3f}')
    return {
        'rccd_mean': float(rccd_mean), 'rccd_std': float(rccd_std),
        'baseline_mean': float(base_mean), 'baseline_std': float(base_std),
        'diff': float(diff),
    }


def latency_pareto_from_sweep(dataset, best_run_name=None, baseline_run_name=None,
                              artifacts_root='artifacts', out_dir='figures',
                              metric='HR@10'):
    teacher_ref = load_teacher_reference(dataset, artifacts_root)
    runs = load_all_sweep_runs(dataset, artifacts_root)
    points = []
    if teacher_ref:
        points.append(('Teacher (NFE=T)',
                       teacher_ref['latency_ms'],
                       teacher_ref.get('metrics_full_nfe_test',
                                       teacher_ref.get('metrics_full_nfe', {})).get(metric, 0),
                       COLORS['teacher'], '*'))
    target_runs = []
    if best_run_name:
        target_runs.append((best_run_name, 'RCCD', COLORS['student'], 'o'))
    if baseline_run_name:
        target_runs.append((baseline_run_name, 'CD baseline', COLORS['baseline'], 's'))
    for run_name, label, color, marker in target_runs:
        r = next((x for x in runs if x['run_name'] == run_name), None)
        if r is None: continue
        for nfe_str, lat_ms in r['latency_per_nfe_ms'].items():
            m = r['test_metrics_per_nfe'][nfe_str][metric]
            points.append((f'{label} NFE={nfe_str}', lat_ms, m, color, marker))
    if not HAS_PLT or not points:
        return points
    apply_style()
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
    _save(fig, os.path.join(out_dir, f'pareto_sweep_{dataset}.pdf'))
    return points


def plot_outlier_boxplot(dataset, run_names, artifacts_root='artifacts',
                         out_dir='figures', labels=None):
    """Per-user rank distribution box-plot. Reveals tail behavior."""
    if not HAS_PLT: return
    apply_style()
    data_per_run = []
    used_labels = []
    for i, rn in enumerate(run_names):
        pred_path = os.path.join(artifacts_root, dataset, rn, 'test_predictions_nfe1.npz')
        if not os.path.exists(pred_path):
            continue
        data = np.load(pred_path)
        ranks = data['target_rank']
        ranks_valid = ranks[ranks >= 0]
        data_per_run.append(ranks_valid)
        used_labels.append(labels[i] if labels else rn)
    if not data_per_run: return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data_per_run, labels=used_labels, showfliers=False,
                    patch_artist=True)
    palette = [COLORS['baseline'], COLORS['student'], COLORS['teacher']]
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(palette[i % len(palette)])
        patch.set_alpha(0.6)
    ax.set_yscale('log')
    ax.set_ylabel('Rank of true target (log scale)')
    ax.set_title(f'Per-user rank distribution ({dataset})')
    _save(fig, os.path.join(out_dir, f'outliers_{dataset}.pdf'))


# ===================================================================
# Section 7: CLI
# ===================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True,
                   choices=['summary', 'latex', 'multi_plots',
                            'model_plots', 'sweep', 'length_aware',
                            'multi_seed_stats', 'pareto_runs',
                            'outliers', 'all'])
    p.add_argument('--dataset', default=None,
                   help='For sweep / length_aware / model_plots / multi_seed_stats / pareto_runs.')
    p.add_argument('--json_paths', nargs='+', default=None,
                   help='Multi-seed JSONs. For summary / latex / multi_plots.')
    p.add_argument('--artifacts_root', default='artifacts')
    p.add_argument('--data_root', default='../datasets/data')
    p.add_argument('--out_dir', default='figures')
    p.add_argument('--metric', default='HR@10')
    p.add_argument('--nfe', default='1')
    p.add_argument('--nfe_grid', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--seed_filter', type=int, default=None)
    p.add_argument('--beta_target', type=float, default=1.0)
    p.add_argument('--tau_target', type=float, default=0.1)
    p.add_argument('--run_name', default=None)
    p.add_argument('--teacher_ckpt', default=None)
    p.add_argument('--student_ckpt', default=None)
    p.add_argument('--trajectory_sample_idx', type=int, default=0)
    p.add_argument('--n_items_tsne', type=int, default=2000)
    p.add_argument('--n_clusters_tsne', type=int, default=8)
    p.add_argument('--svd_top_k', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_latex', default=None)
    p.add_argument('--out_compute_latex', default=None)
    p.add_argument('--logs_root', default='logs')
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Modes requiring multi-seed JSONs ----
    def _need_jsons():
        if not args.json_paths:
            raise SystemExit(f'mode={args.mode} requires --json_paths a.json b.json ...')

    if args.mode in ('summary', 'all'):
        _need_jsons()
        for path in args.json_paths:
            r = load_multiseed_json(path)
            summary_table_text(r, nfe_grid=tuple(args.nfe_grid), metric=args.metric)
            compute_metrics_text(r, nfe_grid=tuple(args.nfe_grid))
            gen_gap_analysis_text(r, nfe_grid=tuple(args.nfe_grid), metric=args.metric)

    if args.mode in ('latex',):
        _need_jsons()
        per_dataset = {}
        for path in args.json_paths:
            r = load_multiseed_json(path)
            per_dataset[r['dataset']] = r
        if args.out_latex:
            Path(args.out_latex).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_latex, 'w') as f:
                f.write(latex_main_table(per_dataset))
            print(f'Main results LaTeX -> {args.out_latex}')
        if args.out_compute_latex:
            Path(args.out_compute_latex).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_compute_latex, 'w') as f:
                f.write(latex_compute_table(per_dataset, nfe_grid=tuple(args.nfe_grid)))
            print(f'Compute LaTeX -> {args.out_compute_latex}')

    if args.mode in ('multi_plots', 'all'):
        _need_jsons()
        per_dataset = {}
        for path in args.json_paths:
            r = load_multiseed_json(path)
            per_dataset[r['dataset']] = r

        plot_tradeoff_nfe_quality(per_dataset, metric=args.metric,
                                  out_path=os.path.join(args.out_dir,
                                  f'tradeoff_{args.metric.replace("@","")}.pdf'))
        plot_tradeoff_nfe_quality(per_dataset, metric='NDCG@10',
                                  out_path=os.path.join(args.out_dir, 'tradeoff_NDCG10.pdf'))
        plot_speedup(per_dataset, out_path=os.path.join(args.out_dir, 'speedup.pdf'))
        plot_main_results_errorbars(per_dataset, metric=args.metric,
                                    out_path=os.path.join(args.out_dir,
                                    'main_results_errorbars.pdf'))
        for name, res in per_dataset.items():
            slug = name.replace('/', '_').replace(' ', '_')
            plot_pareto(res, metric=args.metric,
                        out_path=os.path.join(args.out_dir, f'pareto_{slug}.pdf'))
            plot_latency_bars(res, out_path=os.path.join(args.out_dir, f'latency_{slug}.pdf'))
            plot_bootstrap_ci(res, metric=args.metric,
                              out_path=os.path.join(args.out_dir, f'bootstrap_ci_{slug}.pdf'))
            plot_val_vs_test_gap(res, metric=args.metric,
                                 nfe_grid=tuple(args.nfe_grid),
                                 out_path=os.path.join(args.out_dir, f'val_test_gap_{slug}.pdf'))
            log_dir = os.path.join(args.logs_root, name)
            if os.path.isdir(log_dir):
                plot_convergence(log_dir, out_path=os.path.join(args.out_dir,
                                 f'convergence_{slug}.pdf'))

    if args.mode in ('model_plots',):
        if not args.dataset:
            raise SystemExit('model_plots needs --dataset')
        import torch
        args_ns = _build_args_from_teacher_config(args.dataset, args.data_root,
                                                  args.artifacts_root)
        tckpt = args.teacher_ckpt or os.path.join(args.artifacts_root,
                                                  args.dataset, 'teacher', 'teacher.pt')
        if args.student_ckpt:
            sckpt = args.student_ckpt
        elif args.run_name:
            sckpt = os.path.join(args.artifacts_root, args.dataset,
                                 args.run_name, 'student_final.pt')
        else:
            raise SystemExit('model_plots needs --student_ckpt or --run_name')
        teacher, student, test_loader, _ = _load_models_and_data(args_ns, tckpt, sckpt)
        slug = args.dataset.replace('/', '_')
        suffix = f'_{args.run_name}' if args.run_name else ''
        plot_tsne_embeddings(teacher, student,
                             n_items=args.n_items_tsne, n_clusters=args.n_clusters_tsne,
                             out_path=os.path.join(args.out_dir,
                             f'tsne_embeddings_{slug}{suffix}.pdf'), seed=args.seed)
        plot_denoising_trajectory(teacher, student, test_loader,
                                  torch.device(args_ns.device),
                                  sample_idx=args.trajectory_sample_idx,
                                  out_path=os.path.join(args.out_dir,
                                  f'denoising_trajectory_{slug}{suffix}.pdf'),
                                  seed=args.seed)
        plot_svd_spectrum(teacher, student,
                          out_path=os.path.join(args.out_dir,
                          f'svd_spectrum_{slug}{suffix}.pdf'),
                          top_k=args.svd_top_k)

    if args.mode in ('sweep', 'all'):
        if args.dataset:
            for metric in ['HR@10', 'NDCG@10']:
                sensitivity_analysis(args.dataset, metric=metric, nfe=args.nfe,
                                     artifacts_root=args.artifacts_root,
                                     out_dir=args.out_dir,
                                     seed_filter=args.seed_filter)

    if args.mode == 'length_aware':
        if not args.dataset:
            raise SystemExit('length_aware needs --dataset')
        if args.run_name:
            length_aware_analysis(args.dataset, args.run_name,
                                  artifacts_root=args.artifacts_root,
                                  out_dir=args.out_dir)
        else:
            baseline_rn = f'seed1997_beta0.0_tau0.1'
            best_rn = f'seed1997_beta{args.beta_target}_tau{args.tau_target}'
            length_aware_compare(args.dataset, [baseline_rn, best_rn],
                                 artifacts_root=args.artifacts_root,
                                 out_dir=args.out_dir,
                                 labels=['CD baseline', 'RCCD'])

    if args.mode == 'multi_seed_stats':
        if not args.dataset:
            raise SystemExit('multi_seed_stats needs --dataset')
        multi_seed_statistics_from_sweep(args.dataset,
                                         beta_target=args.beta_target,
                                         tau_target=args.tau_target,
                                         metric=args.metric, nfe=args.nfe,
                                         artifacts_root=args.artifacts_root)

    if args.mode == 'pareto_runs':
        if not args.dataset:
            raise SystemExit('pareto_runs needs --dataset')
        baseline_rn = f'seed1997_beta0.0_tau0.1'
        best_rn = f'seed1997_beta{args.beta_target}_tau{args.tau_target}'
        latency_pareto_from_sweep(args.dataset, best_run_name=best_rn,
                                  baseline_run_name=baseline_rn,
                                  artifacts_root=args.artifacts_root,
                                  out_dir=args.out_dir, metric=args.metric)

    if args.mode == 'outliers':
        if not args.dataset:
            raise SystemExit('outliers needs --dataset')
        baseline_rn = f'seed1997_beta0.0_tau0.1'
        best_rn = f'seed1997_beta{args.beta_target}_tau{args.tau_target}'
        plot_outlier_boxplot(args.dataset, [baseline_rn, best_rn],
                             artifacts_root=args.artifacts_root,
                             out_dir=args.out_dir,
                             labels=['CD baseline', 'RCCD'])


if __name__ == '__main__':
    main()