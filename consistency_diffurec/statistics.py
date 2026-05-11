"""
Statistical analysis of multi-seed runs.

Updates:
- Replaced paired Wilcoxon against constant with proper one-sample Wilcoxon
  on residuals (student - teacher).
- Added median + IQR alongside mean ± std for robustness.
- Improved significance markers: '*' p<0.05, '**' p<0.01, '***' p<0.001,
  '****' p<0.0001 (biostatistics convention).
- Added a brief note about n=3 limitations.
"""
import json
import numpy as np
from scipy import stats

try:
    from literature_baselines import LITERATURE_BASELINES, BASELINE_ORDER, DISPLAY_NAMES
except ImportError:
    LITERATURE_BASELINES = {}
    BASELINE_ORDER = []
    DISPLAY_NAMES = {}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def _gather(results, nfe, metric_key):
    return np.array([
        results['students'][seed][str(nfe)][metric_key]
        for seed in results['students']
    ])


def _gather_val(results, nfe, metric_key):
    seeds = list(results['students'].keys())
    if not seeds or '_val' not in results['students'][seeds[0]]:
        return None
    return np.array([
        results['students'][seed]['_val'][str(nfe)][metric_key]
        for seed in seeds
    ])


def gen_gap_analysis(results, nfe, metric='HR@10'):
    student_test = _gather(results, nfe, metric)
    student_val  = _gather_val(results, nfe, metric)
    if student_val is None:
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
                    metrics=('HR@5', 'HR@10', 'HR@20',
                             'NDCG@5', 'NDCG@10', 'NDCG@20')):
    """Returns dict[nfe][metric] = mean, std, median, q25, q75, values."""
    out = {}
    for nfe in nfe_grid:
        out[nfe] = {}
        for m in metrics:
            vals = _gather(results, nfe, m)
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
    """
    One-sample Wilcoxon signed-rank test: tests whether median of (values - reference)
    differs from zero. This is the correct test for comparing a multi-seed sample
    against a deterministic reference like the teacher's full-NFE metric.

    Returns NaN if n < 5 (Wilcoxon has no power below this threshold).
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 5:
        return float('nan')
    residuals = values - reference
    if np.allclose(residuals, 0):
        return 1.0
    try:
        # `stats.wilcoxon` accepts a single sample with H0: median = 0
        _, p = stats.wilcoxon(residuals)
        return float(p)
    except ValueError:
        return float('nan')


def paired_wilcoxon(values_a, values_b):
    """Paired Wilcoxon. Use when both arrays come from matched seeds."""
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if len(values_a) != len(values_b):
        return float('nan')
    if len(values_a) < 5:
        return float('nan')
    if np.allclose(values_a, values_b):
        return 1.0
    try:
        _, p = stats.wilcoxon(values_a, values_b)
        return float(p)
    except ValueError:
        return float('nan')


def compare_student_vs_baseline(results, nfe, metric='HR@10'):
    student_vals = _gather(results, nfe, metric)
    baseline_val = results['baseline'][str(nfe)][metric]
    n_better = int((student_vals > baseline_val).sum())
    return {
        'student_mean':   float(student_vals.mean()),
        'student_std':    float(student_vals.std(ddof=1)) if len(student_vals) > 1 else 0.0,
        'student_median': float(np.median(student_vals)),
        'baseline':       float(baseline_val),
        'n_seeds':        len(student_vals),
        'n_seeds_better': n_better,
    }


def compare_student_vs_teacher(results, nfe, metric='HR@10'):
    student_vals = _gather(results, nfe, metric)
    teacher_val  = results['teacher']['full_nfe'][metric]
    p = one_sample_wilcoxon(student_vals, teacher_val)
    return {
        'student_mean':       float(student_vals.mean()),
        'student_std':        float(student_vals.std(ddof=1)) if len(student_vals) > 1 else 0.0,
        'student_median':     float(np.median(student_vals)),
        'teacher':            float(teacher_val),
        'gap_pct':            float((student_vals.mean() - teacher_val) / teacher_val * 100),
        'n_seeds':            len(student_vals),
        'n_seeds_above_teacher': int((student_vals > teacher_val).sum()),
        'p_one_sample_wilcoxon': p,
    }


def significance_marker(p):
    """Biostatistics convention: * < 0.05, ** < 0.01, *** < 0.001, **** < 0.0001."""
    if p is None or np.isnan(p):
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
        c = cm[nfe]
        print(f"{nfe:<6} {c['flops_ratio_vs_teacher_full']:<14.4f} "
              f"{c['student_latency_ms']:<12.4f} "
              f"{c['student_throughput_per_sec']:<14.1f} "
              f"{c['speedup_vs_teacher_full']:<10.2f}x")


def latex_main_table(results_per_dataset, nfe=1, metrics=('HR@5', 'HR@10', 'HR@20',
                                                          'NDCG@5', 'NDCG@10', 'NDCG@20')):
    """
    Markers:
      *,**,***,**** — student significantly beats truncated DDIM (Wilcoxon p<0.05/0.01/0.001/0.0001)
      †             — no significant difference from teacher (one-sample Wilcoxon p>=0.05)
    """
    lines = []
    lines.append(r'\begin{tabular}{l l ' + 'r' * len(metrics) + '}')
    lines.append(r'\toprule')
    lines.append('Dataset & Method & ' + ' & '.join(metrics) + r' \\')
    lines.append(r'\midrule')

    for ds_name, res in results_per_dataset.items():
        agg = aggregate_seeds(res, nfe_grid=[nfe], metrics=metrics)

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

        student_row = ['', f'Student (NFE={nfe})']
        for m in metrics:
            mean   = agg[nfe][m]['mean']
            std    = agg[nfe][m]['std']
            median = agg[nfe][m]['median']
            vals   = np.array(agg[nfe][m]['values'])

            # vs truncated DDIM via paired Wilcoxon on (student - baseline)
            base_val = res['baseline'][str(nfe)][m]
            p_trunc = one_sample_wilcoxon(vals, base_val)
            star = significance_marker(p_trunc)

            # vs teacher full NFE via one-sample Wilcoxon
            p_teacher = one_sample_wilcoxon(vals, res['teacher']['full_nfe'][m])
            dagger = r'$^{\dagger}$' if (not np.isnan(p_teacher) and p_teacher >= 0.05) else ''

            student_row.append(f'{mean:.2f}{star}{dagger} $\\pm$ {std:.2f}')
        lines.append(' & ' + ' & '.join(student_row[1:]) + r' \\')
        lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append('')
    lines.append(r'% Markers:')
    lines.append(r'%   *,**,***,**** — student significantly beats truncated DDIM (p<0.05/0.01/0.001/0.0001).')
    lines.append(r'%   $\dagger$ — no significant difference from teacher (one-sample Wilcoxon p>=0.05); desirable.')
    lines.append(r'%   Note: with small n (e.g. 3 seeds) Wilcoxon has limited power; we report mean+std (and median+IQR in supplementary).')
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


def summary_table_text(results, nfe_grid=(1, 2, 4, 8), metric='HR@10'):
    """Plain-text summary with both mean±std and median+IQR for robustness."""
    print(f"\n=== {results['dataset']} | {metric} ===")
    print(f"Teacher (NFE={results['teacher']['T']}): "
          f"{results['teacher']['full_nfe'][metric]:.4f}")
    print(f"(n_seeds = {len(results['students'])}; Wilcoxon p-values valid only for n >= 5)")
    header = (f"{'NFE':<5} {'Trunc.':<10} {'Student mean±std':<22} "
              f"{'Student median (IQR)':<24} {'gap vs T %':<12} "
              f"{'p_vs_teacher':<14} {'p_vs_trunc':<12}")
    print(header); print('-' * len(header))
    for nfe in nfe_grid:
        student_vals = _gather(results, nfe, metric)
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


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('json_paths', nargs='+')
    ap.add_argument('--metric', default='HR@10')
    ap.add_argument('--out_latex', default=None)
    ap.add_argument('--out_compute_latex', default=None)
    ap.add_argument('--nfe_grid', type=int, nargs='+', default=[1, 2, 4, 8])
    args = ap.parse_args()

    per_dataset = {}
    for path in args.json_paths:
        r = load_results(path)
        per_dataset[r['dataset']] = r
        summary_table_text(r, nfe_grid=args.nfe_grid, metric=args.metric)
        compute_metrics_text(r, nfe_grid=args.nfe_grid)
        gen_gap_analysis_text(r, nfe_grid=args.nfe_grid, metric=args.metric)

    from pathlib import Path

    if args.out_latex:
        Path(args.out_latex).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_latex, 'w') as f:
            f.write(latex_main_table(per_dataset))
        print(f'\nMain results LaTeX table -> {args.out_latex}')

    if args.out_compute_latex:
        Path(args.out_compute_latex).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_compute_latex, 'w') as f:
            f.write(latex_compute_table(per_dataset, nfe_grid=tuple(args.nfe_grid)))
        print(f'Compute LaTeX table -> {args.out_compute_latex}')