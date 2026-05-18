"""
Unified analysis script — Colab/notebook edition.

All artifacts go INLINE into the notebook cell output. No PDFs, no PNGs, no
LaTeX tables saved to disk. Style: white background, no grid lines.

Statistical methodology:
  - 5 evaluation seeds → mean ± std, plus bootstrap 95% CI in main tables
  - Effect size (Cohen's d) reported alongside p-values, since n=5 makes
    Wilcoxon underpowered
  - Holm-Bonferroni correction for multiple pairwise comparisons across
    datasets × NFE × methods
  - Significance markers limited to */**/*** (ML convention, no **** as in
    biostatistics)
  - Note: teacher trained on a single seed (no variance available); reported
    student-vs-teacher comparisons therefore use one-sample Wilcoxon as a
    conservative test (documented in tables)

CLI modes:
  --mode summary          stdout summary + compute & val/test gap
  --mode plots            inline plots (tradeoff, pareto, latency, speedup, …)
  --mode sweep            sensitivity grid + per-config tables (text)
  --mode length_aware     per-bin HR@10 (inline plot)
  --mode multi_seed_stats RCCD vs CD-only baseline (paired, with Cohen's d)
  --mode all              run main analyses end-to-end
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


# ===================================================================
# Section 0: shared inline-friendly style
# ===================================================================

COLORS = {
    'student':    '#1a1a1a',
    'teacher':    '#DAA520',
    'baseline':   '#8B4513',
    'tertiary':   '#B8860B',
    'fill_light': '#F4E4A1',
}
MARKERS = {'student': 'o', 'teacher': 's', 'baseline': '^', 'tertiary': 'D'}


def apply_style():
    """White background, no grid. Suitable for inline notebook display."""
    if not HAS_PLT:
        return
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
        'axes.grid':           False,  # no grid per requirements
        'lines.linewidth':     2.0,
        'lines.markersize':    7,
    })


def _show(fig):
    """Display figure inline. Never save to disk."""
    if not HAS_PLT:
        return
    plt.show()
    plt.close(fig)


# ===================================================================
# Section 1: loaders
# ===================================================================

def load_multiseed_json(path):
    with open(path) as f:
        return json.load(f)


def load_all_sweep_runs(dataset, artifacts_root):
    """Load every per-run summary.json under artifacts/<dataset>/, excluding teacher/."""
    runs = []
    pattern = os.path.join(artifacts_root, dataset, '*', 'summary.json')
    for path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(path)
        run_name = os.path.basename(run_dir)
        if run_name == 'teacher':
            continue
        with open(path) as f:
            summary = json.load(f)
        cfg_path = os.path.join(run_dir, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                summary['config'] = json.load(f)
        else:
            summary['config'] = {}
        summary['run_dir'] = run_dir
        runs.append(summary)
    return runs


# ===================================================================
# Section 2: statistical primitives
# ===================================================================

def _gather(results, nfe, metric_key, students_key='students'):
    out = []
    for _, payload in results[students_key].items():
        if not isinstance(payload, dict) or str(nfe) not in payload:
            continue
        out.append(payload[str(nfe)][metric_key])
    return np.array(out)


def _gather_val(results, nfe, metric_key, students_key='students'):
    out = []
    for _, payload in results[students_key].items():
        if not isinstance(payload, dict) or '_val' not in payload:
            continue
        if str(nfe) not in payload['_val']:
            continue
        out.append(payload['_val'][str(nfe)][metric_key])
    return np.array(out) if out else None


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def one_sample_wilcoxon(values, reference):
    """One-sample Wilcoxon test. Conservative when reference has no variance."""
    values = np.asarray(values, dtype=float)
    if len(values) < 5 or not HAS_SCIPY:
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
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b) or len(a) < 5 or not HAS_SCIPY:
        return float('nan')
    if np.allclose(a, b):
        return 1.0
    try:
        _, p = sst.wilcoxon(a, b)
        return float(p)
    except (ValueError, AttributeError):
        return float('nan')


def cohens_d_paired(a, b):
    """Paired Cohen's d. For 5 evaluation seeds where both methods see the same seed."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diff) < 2:
        return float('nan')
    sd = np.std(diff, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(diff.mean() / sd)


def cohens_d_independent(a, b):
    """Independent-samples Cohen's d. Use when sample sizes differ or seeds independent."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1)
                      + (len(b) - 1) * np.var(b, ddof=1))
                     / (len(a) + len(b) - 2))
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def interpret_cohens_d(d):
    """Standard interpretation (Cohen 1988)."""
    a = abs(d)
    if a < 0.2: return 'negligible'
    if a < 0.5: return 'small'
    if a < 0.8: return 'medium'
    return 'large'


def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    adj = np.empty(n)
    prev = 0.0
    for rank, idx in enumerate(order):
        p_adj = pvals[idx] * (n - rank)
        p_adj = min(1.0, max(p_adj, prev))  # monotonic
        adj[idx] = p_adj
        prev = p_adj
    return adj


def significance_marker(p):
    """ML convention: * <0.05, ** <0.01, *** <0.001. No **** (biostat-only)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    if p < 0.001: return r'$^{***}$'
    if p < 0.01:  return r'$^{**}$'
    if p < 0.05:  return r'$^{*}$'
    return ''


# ===================================================================
# Section 3: aggregation
# ===================================================================

def aggregate_seeds(results, nfe_grid, metrics, students_key='students',
                    with_ci=True, n_boot=10000):
    out = {}
    for nfe in nfe_grid:
        out[nfe] = {}
        for m in metrics:
            vals = _gather(results, nfe, m, students_key)
            if len(vals) == 0:
                out[nfe][m] = {'mean': 0, 'std': 0, 'ci_lo': 0, 'ci_hi': 0,
                               'values': []}
                continue
            entry = {
                'mean':   float(vals.mean()),
                'std':    float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                'values': vals.tolist(),
            }
            if with_ci and len(vals) > 1:
                lo, hi = bootstrap_ci(vals, n_boot=n_boot)
                entry['ci_lo'] = lo; entry['ci_hi'] = hi
            else:
                entry['ci_lo'] = entry['mean']; entry['ci_hi'] = entry['mean']
            out[nfe][m] = entry
    return out


# ===================================================================
# Section 4: text reports (NO LaTeX)
# ===================================================================

def summary_table_text(results, nfe_grid, metric='HR@10', apply_holm=True):
    """Main results: mean [95% CI] + p (vs teacher, conservative) + Cohen's d.

    All p-values across NFEs collected, then Holm-Bonferroni corrected.
    """
    print(f"\n=== {results['dataset']} | {metric} ===")
    n_seeds = sum(1 for k, v in results['students'].items() if isinstance(v, dict))
    teacher_val = results['teacher']['full_nfe'][metric]
    print(f"Teacher (NFE={results['teacher']['T']}): {teacher_val:.4f}  "
          f"[single-seed, no variance — see Notes]")
    print(f"n_seeds = {n_seeds}\n")

    # First pass: gather raw p-values per NFE
    raw_p_teacher = []
    raw_p_baseline = []
    has_cd_baseline = 'students_baseline' in results and results['students_baseline']
    rows = []
    for nfe in nfe_grid:
        s = _gather(results, nfe, metric)
        if len(s) == 0:
            continue
        p_t = one_sample_wilcoxon(s, teacher_val)
        raw_p_teacher.append(p_t)
        row = {'nfe': nfe, 'student': s, 'p_teacher': p_t}
        if has_cd_baseline:
            cd = _gather(results, nfe, metric, students_key='students_baseline')
            if len(cd) == len(s):
                row['cd_baseline'] = cd
                p_cd = paired_wilcoxon(s, cd)
                row['p_vs_cd'] = p_cd
                raw_p_baseline.append(p_cd)
                row['d_vs_cd'] = cohens_d_paired(s, cd)
            else:
                row['cd_baseline'] = None
                row['p_vs_cd'] = float('nan')
                row['d_vs_cd'] = float('nan')
        rows.append(row)

    # Holm-Bonferroni correction across NFEs for each comparison family
    if apply_holm and raw_p_teacher:
        p_teacher_adj = holm_bonferroni(raw_p_teacher)
        for r, p_adj in zip(rows, p_teacher_adj):
            r['p_teacher_adj'] = p_adj
    if apply_holm and raw_p_baseline:
        p_cd_adj = holm_bonferroni(raw_p_baseline)
        adj_iter = iter(p_cd_adj)
        for r in rows:
            if r.get('cd_baseline') is not None:
                r['p_vs_cd_adj'] = next(adj_iter)

    # Print
    if has_cd_baseline:
        hdr = (f"{'NFE':<5} {'Student mean [95% CI]':<28} "
               f"{'CD-only mean':<14} "
               f"{'p_vs_teacher':<14} {'p_vs_CD':<10} {'Cohen d':<14}")
    else:
        hdr = (f"{'NFE':<5} {'Student mean [95% CI]':<28} "
               f"{'p_vs_teacher (adj)':<20}")
    print(hdr); print('-' * len(hdr))

    for r in rows:
        s = r['student']
        m_mean = s.mean()
        ci_lo, ci_hi = bootstrap_ci(s)
        p_t_adj = r.get('p_teacher_adj', r['p_teacher'])
        p_t_str = f"{p_t_adj:.4g}{significance_marker(p_t_adj)}".replace(' ', '')

        student_cell = f"{m_mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]"
        if has_cd_baseline and r.get('cd_baseline') is not None:
            cd = r['cd_baseline']
            cd_cell = f"{cd.mean():.4f}"
            p_cd_adj = r.get('p_vs_cd_adj', r['p_vs_cd'])
            p_cd_str = (f"{p_cd_adj:.4g}{significance_marker(p_cd_adj)}"
                        if not np.isnan(p_cd_adj) else 'n/a')
            d = r['d_vs_cd']
            d_str = (f"{d:+.2f} ({interpret_cohens_d(d)})"
                     if not np.isnan(d) else 'n/a')
            print(f"{r['nfe']:<5} {student_cell:<28} {cd_cell:<14} "
                  f"{p_t_str:<14} {p_cd_str:<10} {d_str:<14}")
        else:
            print(f"{r['nfe']:<5} {student_cell:<28} {p_t_str:<20}")

    print("\nNotes:")
    print("  - Teacher trained on a single seed; p_vs_teacher uses one-sample")
    print("    Wilcoxon (conservative). Significance markers use ML convention")
    print("    (* p<0.05, ** p<0.01, *** p<0.001).")
    if apply_holm:
        print("  - p-values Holm-Bonferroni corrected across NFEs within each family.")
    print("  - 95% CI from 10k bootstrap resamples; Cohen's d on paired seed differences.")


def compute_metrics(results, nfe_grid):
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
            'flops_ratio':     nfe / T,
            's_throughput':    1000.0 / s_lat,
            't_throughput':    1000.0 / t_lat,
            's_latency_ms':    s_lat,
            't_latency_ms':    t_lat,
            'speedup':         teacher_full_ms / s_lat,
        }
    out['_teacher_full'] = {
        'latency_ms': teacher_full_ms,
        'throughput': 1000.0 / teacher_full_ms,
        'T': T,
    }
    return out


def compute_metrics_text(results, nfe_grid):
    cm = compute_metrics(results, nfe_grid)
    tf = cm['_teacher_full']
    print(f"\n=== {results['dataset']} | compute & speed ===")
    print(f"Teacher full (NFE={tf['T']}): {tf['latency_ms']:.4f} ms, "
          f"{tf['throughput']:.1f} samples/s")
    hdr = (f"{'NFE':<6} {'FLOPs ratio':<14} {'Student ms':<12} "
           f"{'Throughput':<14} {'Speedup':<10}")
    print(hdr); print('-' * len(hdr))
    for nfe in nfe_grid:
        if nfe not in cm: continue
        c = cm[nfe]
        print(f"{nfe:<6} {c['flops_ratio']:<14.4f} "
              f"{c['s_latency_ms']:<12.4f} {c['s_throughput']:<14.1f} "
              f"{c['speedup']:<10.2f}x")


def latency_table_text(results_per_dataset, nfe=1):
    """Latency comparison table — text-only, inline output.

    Shows: Method | Steps | Latency (ms) | Speedup vs DiffuRec | HR@10
    Only DiffuRec teacher (NFE=T) and RCCD student (NFE=nfe). No other baselines.
    """
    print(f"\n=== Latency table (NFE={nfe}, batch_size=1, 1000 inferences) ===\n")
    hdr = (f"{'Dataset':<14} {'Method':<14} {'Steps':<7} "
           f"{'Latency (ms)':<14} {'Speedup':<10} {'HR@10':<8}")
    print(hdr); print('-' * len(hdr))
    for ds, res in results_per_dataset.items():
        T = res['teacher']['T']
        t_lat = res['latency']['teacher_full']
        t_hr = res['teacher']['full_nfe']['HR@10']
        print(f"{ds:<14} {'DiffuRec':<14} {T:<7} "
              f"{t_lat:<14.4f} {'1.00x':<10} {t_hr:<8.4f}")
        s_lat = res['latency']['student'][str(nfe)]
        s_hr_vals = _gather(res, nfe, 'HR@10')
        s_hr = s_hr_vals.mean() if len(s_hr_vals) else 0.0
        speedup = t_lat / s_lat if s_lat > 0 else float('inf')
        print(f"{'':<14} {'RCCD (ours)':<14} {nfe:<7} "
              f"{s_lat:<14.4f} {speedup:<10.2f} {s_hr:<8.4f}")
        print('-' * len(hdr))


def gen_gap_analysis_text(results, nfe_grid, metric='HR@10'):
    print(f"\n=== {results['dataset']} | val/test gap | {metric} ===")
    teacher_test = results['teacher']['full_nfe'].get(metric)
    teacher_val  = results['teacher'].get('full_nfe_val', {}).get(metric)
    if teacher_val is not None:
        print(f"Teacher: val={teacher_val:.4f}  test={teacher_test:.4f}  "
              f"gap={teacher_val - teacher_test:+.4f}")
    hdr = f"{'NFE':<5} {'Student val':<22} {'Student test':<22} {'Gap':<10}"
    print(hdr); print('-' * len(hdr))
    for nfe in nfe_grid:
        s_test = _gather(results, nfe, metric)
        s_val = _gather_val(results, nfe, metric)
        if s_val is None or len(s_val) == 0 or len(s_test) == 0:
            continue
        gap = s_val.mean() - s_test.mean()
        print(f"{nfe:<5} "
              f"{s_val.mean():.4f} ± {s_val.std(ddof=1) if len(s_val)>1 else 0:.4f}     "
              f"{s_test.mean():.4f} ± {s_test.std(ddof=1) if len(s_test)>1 else 0:.4f}     "
              f"{gap:+.4f}")


def final_val_metric_text(results, nfe=1, metric='HR@10'):
    """Per-seed final val metric (replaces convergence curves).

    Reports best val HR@10 for each seed, for RCCD and (if present) CD-only.
    """
    print(f"\n=== {results['dataset']} | best val {metric} @ NFE={nfe} per seed ===")
    s_val = _gather_val(results, nfe, metric, students_key='students')
    if s_val is not None and len(s_val) > 0:
        print(f"RCCD:    mean={s_val.mean():.4f}, std={s_val.std(ddof=1) if len(s_val)>1 else 0:.4f}, "
              f"per-seed={[f'{v:.4f}' for v in s_val.tolist()]}")
    if 'students_baseline' in results and results['students_baseline']:
        cd_val = _gather_val(results, nfe, metric, students_key='students_baseline')
        if cd_val is not None and len(cd_val) > 0:
            print(f"CD-only: mean={cd_val.mean():.4f}, std={cd_val.std(ddof=1) if len(cd_val)>1 else 0:.4f}, "
                  f"per-seed={[f'{v:.4f}' for v in cd_val.tolist()]}")


# ===================================================================
# Section 5: inline plots (no save)
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


def plot_tradeoff_nfe_quality(results_per_dataset, metric='HR@10'):
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
        means, stds = _student_mean_std(res, nfe_grid, metric)
        ax.errorbar(nfe_grid, means, yerr=stds, marker=MARKERS['student'],
                    color=COLORS['student'], linewidth=2.0, capsize=4,
                    label='RCCD student')
        ax.fill_between(nfe_grid, means - stds, means + stds,
                        color=COLORS['fill_light'], alpha=0.4)
        ax.set_xscale('log', base=2)
        ax.set_xticks(nfe_grid); ax.set_xticklabels(nfe_grid)
        ax.set_xlabel('NFE'); ax.set_ylabel(metric)
        ax.set_title(ds_name)
        ax.legend(loc='lower right', framealpha=0.95)
    fig.suptitle(f'Quality vs inference cost ({metric})', y=1.02, fontsize=13)
    fig.tight_layout()
    _show(fig)


def plot_pareto(results, metric='HR@10'):
    if not HAS_PLT: return
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    nfe_grid = sorted(int(k) for k in results['latency']['student'].keys())
    s_means, s_stds = _student_mean_std(results, nfe_grid, metric)
    s_lat = [results['latency']['student'][str(n_)] for n_ in nfe_grid]
    ax.errorbar(s_lat, s_means, yerr=s_stds, marker=MARKERS['student'],
                color=COLORS['student'], linewidth=2.0, capsize=4,
                label='RCCD student')
    for nfe, x_, y_ in zip(nfe_grid, s_lat, s_means):
        ax.annotate(f'NFE={nfe}', (x_, y_), textcoords='offset points',
                    xytext=(6, 6), fontsize=8, color=COLORS['student'])
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
    fig.tight_layout()
    _show(fig)


def plot_latency_bars(results):
    if not HAS_PLT: return
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    nfe_grid = sorted(int(k) for k in results['latency']['student'].keys())
    x = np.arange(len(nfe_grid))
    s_lat = [results['latency']['student'][str(n_)] for n_ in nfe_grid]
    ax.bar(x, s_lat, 0.6, color=COLORS['student'], label='RCCD student',
           edgecolor='black', linewidth=0.5)
    ax.axhline(results['latency']['teacher_full'], color=COLORS['teacher'],
               linestyle=':', linewidth=1.8,
               label=f"Teacher full (NFE={results['teacher']['T']})")
    ax.set_xticks(x)
    ax.set_xticklabels([f'NFE={n_}' for n_ in nfe_grid])
    ax.set_ylabel('Latency (ms / sample)')
    ax.set_yscale('log')
    ax.set_title(f'Inference latency — {results["dataset"]}')
    ax.legend()
    fig.tight_layout()
    _show(fig)


def plot_speedup(results_per_dataset):
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
        speedups = [r['latency']['teacher_full'] / r['latency']['student'][str(nfe)]
                    for r in results_per_dataset.values()]
        ax.bar(x + (i - len(nfe_grid[:4]) / 2 + 0.5) * w, speedups, w,
               label=f'NFE={nfe}', color=cmap[i % len(cmap)],
               edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel(r'Speedup vs teacher full ($\times$)')
    ax.set_title('Inference speedup of distilled student')
    ax.legend()
    fig.tight_layout()
    _show(fig)


def plot_main_results_errorbars(results_per_dataset, metric='HR@10'):
    """RCCD vs Teacher only. No other baselines per requirements."""
    if not HAS_PLT: return
    apply_style()
    datasets = list(results_per_dataset.keys())
    n = len(datasets)
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * n), 4.5))
    x = np.arange(n); w = 0.35
    teacher_vals = [r['teacher']['full_nfe'][metric] for r in results_per_dataset.values()]
    student_means, student_stds = [], []
    for r in results_per_dataset.values():
        v = _gather(r, 1, metric)
        student_means.append(v.mean() if len(v) else 0)
        student_stds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
    ax.bar(x - w / 2, teacher_vals, w, color=COLORS['teacher'],
           label='Teacher (NFE=T)', edgecolor='black', linewidth=0.5)
    ax.bar(x + w / 2, student_means, w, yerr=student_stds, capsize=5,
           color=COLORS['student'], label='RCCD student (NFE=1)',
           edgecolor='black', linewidth=0.5,
           error_kw={'ecolor': COLORS['tertiary'], 'elinewidth': 1.5})
    ax.set_xticks(x); ax.set_xticklabels(datasets); ax.set_ylabel(metric)
    ax.set_title(f'Main results ({metric}, mean $\\pm$ std)')
    ax.legend()
    fig.tight_layout()
    _show(fig)


def plot_bootstrap_ci(results, metric='HR@10', n_boot=10000):
    if not HAS_PLT: return
    apply_style()
    nfe_grid = sorted(int(k) for k in results['baseline'].keys())
    means, los, his = [], [], []
    rng = np.random.default_rng(0)
    for nfe in nfe_grid:
        v = _gather(results, nfe, metric)
        if len(v) == 0:
            means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
        means.append(v.mean())
        if len(v) > 1:
            boots = np.array([rng.choice(v, size=len(v), replace=True).mean()
                              for _ in range(n_boot)])
            los.append(np.percentile(boots, 2.5))
            his.append(np.percentile(boots, 97.5))
        else:
            los.append(v.mean()); his.append(v.mean())
    means = np.array(means); los = np.array(los); his = np.array(his)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(nfe_grid, means, marker=MARKERS['student'], color=COLORS['student'],
            linewidth=2.0, label='Student (mean)')
    ax.fill_between(nfe_grid, los, his, color=COLORS['fill_light'], alpha=0.6,
                    label='95% bootstrap CI')
    ax.axhline(results['teacher']['full_nfe'][metric], color=COLORS['teacher'],
               linestyle=':', linewidth=1.8,
               label=f"Teacher full (NFE={results['teacher']['T']})")
    ax.set_xscale('log', base=2)
    ax.set_xticks(nfe_grid); ax.set_xticklabels(nfe_grid)
    ax.set_xlabel('NFE'); ax.set_ylabel(metric)
    ax.set_title(f'95% bootstrap CI — {results["dataset"]}')
    ax.legend()
    fig.tight_layout()
    _show(fig)


def plot_val_vs_test_gap(results, metric='HR@10', nfe_grid=(1, 2, 4, 8)):
    if not HAS_PLT: return
    apply_style()
    val_means, val_stds, test_means, test_stds, nfe_used = [], [], [], [], []
    for nfe in nfe_grid:
        t_v = _gather(results, nfe, metric)
        v_v = _gather_val(results, nfe, metric)
        if v_v is None or len(v_v) == 0 or len(t_v) == 0:
            continue
        nfe_used.append(nfe)
        val_means.append(v_v.mean())
        val_stds.append(v_v.std(ddof=1) if len(v_v) > 1 else 0.0)
        test_means.append(t_v.mean())
        test_stds.append(t_v.std(ddof=1) if len(t_v) > 1 else 0.0)
    if not nfe_used:
        return
    val_means = np.array(val_means); val_stds = np.array(val_stds)
    test_means = np.array(test_means); test_stds = np.array(test_stds)
    x = np.arange(len(nfe_used)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w / 2, val_means, w, yerr=val_stds, capsize=3,
           color=COLORS['student'], alpha=0.85, edgecolor='black',
           linewidth=0.5, label='Student — Val')
    ax.bar(x + w / 2, test_means, w, yerr=test_stds, capsize=3,
           color=COLORS['fill_light'], alpha=0.95, edgecolor='black',
           linewidth=0.5, label='Student — Test')
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in nfe_used])
    ax.set_xlabel('NFE'); ax.set_ylabel(metric)
    ax.set_title(f'Val vs test {metric} — {results["dataset"]}')
    ax.legend()
    fig.tight_layout()
    _show(fig)


# ===================================================================
# Section 6: sweep analyses (text + inline heatmap)
# ===================================================================

def sensitivity_analysis(dataset, artifacts_root, metric='HR@10', nfe='1',
                         seed_filter=None):
    runs = load_all_sweep_runs(dataset, artifacts_root)
    if seed_filter is not None:
        runs = [r for r in runs if r.get('random_seed') == seed_filter]
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
        print(f'\nBest: beta={betas[best_idx[0]]}, tau={taus[best_idx[1]]} '
              f'-> {metric}={matrix[best_idx]:.4f}')
    if HAS_PLT:
        apply_style()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(matrix, aspect='auto', cmap='cividis')
        ax.set_xticks(range(len(taus))); ax.set_xticklabels([f'{t}' for t in taus])
        ax.set_yticks(range(len(betas))); ax.set_yticklabels([f'{b}' for b in betas])
        ax.set_xlabel(r'Contrast temperature $\tau$')
        ax.set_ylabel(r'Contrast weight $\beta$')
        ax.set_title(f'{metric} @ NFE={nfe} ({dataset})')
        for i in range(len(betas)):
            for j in range(len(taus)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            color='white' if v < np.nanmean(matrix) else 'black',
                            fontsize=10)
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        _show(fig)
    return matrix, betas, taus


# ===================================================================
# Section 7: multi-seed paired stats (RCCD vs CD-only ablation)
# ===================================================================

def multi_seed_statistics(results, metric='HR@10', nfe='1', n_boot=10000):
    """Paired RCCD vs CD-only ablation from a single multiseed JSON.

    Uses paired Wilcoxon + paired Cohen's d. Per requirement: report effect
    size since Wilcoxon on n=5 is underpowered.
    """
    if 'students_baseline' not in results or not results['students_baseline']:
        print('No CD-only baseline in this JSON — skipping ablation stats.')
        return None
    rccd = _gather(results, int(nfe), metric, students_key='students')
    cd = _gather(results, int(nfe), metric, students_key='students_baseline')
    if len(rccd) == 0 or len(cd) == 0:
        print('Empty seed groups; skipping.')
        return None
    print(f'\n=== Paired ablation: {results["dataset"]}, {metric}@NFE={nfe} ===')
    print(f"RCCD:    n={len(rccd)}, mean={rccd.mean():.4f}, "
          f"std={rccd.std(ddof=1) if len(rccd)>1 else 0:.4f}, vals={rccd.tolist()}")
    print(f"CD-only: n={len(cd)}, mean={cd.mean():.4f}, "
          f"std={cd.std(ddof=1) if len(cd)>1 else 0:.4f}, vals={cd.tolist()}")

    if len(rccd) == len(cd):
        p = paired_wilcoxon(rccd, cd)
        d = cohens_d_paired(rccd, cd)
        print(f'Paired Wilcoxon p = {p:.4g} (n=5 ⇒ underpowered; rely on effect size)')
        print(f"Paired Cohen's d = {d:+.3f} ({interpret_cohens_d(d)})")
        # Bootstrap CI on the mean difference
        diff = rccd - cd
        rng = np.random.default_rng(0)
        boots = np.array([rng.choice(diff, len(diff), replace=True).mean()
                          for _ in range(n_boot)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f'Mean diff = {diff.mean():+.4f}, '
              f'bootstrap 95% CI = [{lo:+.4f}, {hi:+.4f}]')
        print(f'P(RCCD > CD-only across resamples) = {(boots > 0).mean():.3f}')
        return {
            'rccd_mean': float(rccd.mean()), 'cd_mean': float(cd.mean()),
            'diff_mean': float(diff.mean()), 'ci_lo': float(lo), 'ci_hi': float(hi),
            'p_wilcoxon': float(p), 'cohens_d': float(d),
        }


# ===================================================================
# Section 8: length-aware analysis (single dataset)
# ===================================================================

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0, center - spread), min(1, center + spread))


def length_aware_compare(dataset, artifacts_root, run_names, labels=None, n_bins=5):
    all_results = {}
    for rn in run_names:
        run_dir = os.path.join(artifacts_root, dataset, rn)
        pred_path = os.path.join(run_dir, 'test_predictions_nfe1.npz')
        if not os.path.exists(pred_path):
            print(f'[length] missing {pred_path}'); continue
        data = np.load(pred_path)
        lengths = data['hist_lengths']
        ks = data['ks']
        hits = data['hit_at_k']
        try:
            k10_idx = list(ks).index(10)
        except ValueError:
            k10_idx = 1
        quantiles = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
        bin_results = []
        for i in range(n_bins):
            lo, hi = quantiles[i], quantiles[i + 1]
            mask = (lengths >= lo) & (lengths <= hi if i == n_bins - 1 else lengths < hi)
            n = int(mask.sum())
            if n == 0: continue
            hr = float(hits[mask, k10_idx].mean()) * 100
            k_hits = int(hits[mask, k10_idx].sum())
            ci_lo, ci_hi = wilson_ci(k_hits, n)
            bin_results.append({
                'bin': i + 1, 'len_lo': int(lo), 'len_hi': int(hi),
                'n': n, 'HR@10': hr,
                'ci_lo': ci_lo * 100, 'ci_hi': ci_hi * 100,
            })
        all_results[rn] = bin_results
        print(f'\n=== Length-aware: {dataset} / {rn} ===')
        print(f'{"Bin":<5} {"Range":<15} {"n":<6} {"HR@10":<8} {"95% Wilson CI":<20}')
        for r in bin_results:
            print(f'{r["bin"]:<5} {r["len_lo"]}-{r["len_hi"]:<10} {r["n"]:<6} '
                  f'{r["HR@10"]:<8.2f} [{r["ci_lo"]:.2f}, {r["ci_hi"]:.2f}]')

    if HAS_PLT and len(all_results) >= 2:
        apply_style()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        runs_list = list(all_results.keys())
        bins_x = [f'{r["len_lo"]}-{r["len_hi"]}' for r in all_results[runs_list[0]]]
        width = 0.8 / len(runs_list); x = np.arange(len(bins_x))
        palette = [COLORS['baseline'], COLORS['student'], COLORS['teacher']]
        for i, rn in enumerate(runs_list):
            ys = [r['HR@10'] for r in all_results[rn]]
            lab = labels[i] if labels else rn
            ax.bar(x + (i - len(runs_list) / 2 + 0.5) * width, ys, width,
                   color=palette[i % len(palette)], edgecolor='black',
                   linewidth=0.5, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(bins_x)
        ax.set_xlabel('History length range'); ax.set_ylabel('HR@10 (%)')
        ax.set_title(f'Length-aware comparison ({dataset})'); ax.legend()
        fig.tight_layout()
        _show(fig)
    return all_results


# ===================================================================
# Section 9: CLI
# ===================================================================

DRIVE_BASE = '/content/drive/MyDrive/diffurec-distillation-results/consistency-diffurec-after-sweep'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True,
                   choices=['summary', 'plots', 'sweep', 'length_aware',
                            'multi_seed_stats', 'latency', 'all'])
    p.add_argument('--dataset', default=None)
    p.add_argument('--json_paths', nargs='+', default=None)
    p.add_argument('--artifacts_root', default=f'{DRIVE_BASE}/artifacts')
    p.add_argument('--metric', default='HR@10')
    p.add_argument('--nfe', default='1')
    p.add_argument('--nfe_grid', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--beta_target', type=float, default=2.0)
    p.add_argument('--tau_target', type=float, default=0.1)
    p.add_argument('--seed_filter', type=int, default=None)
    p.add_argument('--run_name', default=None)
    args = p.parse_args()

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
            final_val_metric_text(r, nfe=1, metric=args.metric)

    if args.mode in ('plots', 'all'):
        _need_jsons()
        per_dataset = {}
        for path in args.json_paths:
            r = load_multiseed_json(path)
            per_dataset[r['dataset']] = r
        plot_tradeoff_nfe_quality(per_dataset, metric=args.metric)
        plot_speedup(per_dataset)
        plot_main_results_errorbars(per_dataset, metric=args.metric)
        for _, res in per_dataset.items():
            plot_pareto(res, metric=args.metric)
            plot_latency_bars(res)
            plot_bootstrap_ci(res, metric=args.metric)
            plot_val_vs_test_gap(res, metric=args.metric,
                                 nfe_grid=tuple(args.nfe_grid))

    if args.mode == 'latency':
        _need_jsons()
        per_dataset = {}
        for path in args.json_paths:
            r = load_multiseed_json(path)
            per_dataset[r['dataset']] = r
        latency_table_text(per_dataset, nfe=int(args.nfe))

    if args.mode == 'multi_seed_stats':
        _need_jsons()
        for path in args.json_paths:
            r = load_multiseed_json(path)
            multi_seed_statistics(r, metric=args.metric, nfe=args.nfe)

    if args.mode == 'sweep':
        if not args.dataset:
            raise SystemExit('sweep needs --dataset')
        for metric in ['HR@10', 'NDCG@10']:
            sensitivity_analysis(args.dataset, args.artifacts_root,
                                 metric=metric, nfe=args.nfe,
                                 seed_filter=args.seed_filter)

    if args.mode == 'length_aware':
        if not args.dataset:
            raise SystemExit('length_aware needs --dataset')
        baseline_rn = f'seed1907_beta0.0_tau{args.tau_target}_baseline'
        best_rn = f'seed1907_beta{args.beta_target}_tau{args.tau_target}'
        length_aware_compare(args.dataset, args.artifacts_root,
                             [baseline_rn, best_rn],
                             labels=['CD-only', 'RCCD'])


if __name__ == '__main__':
    main()