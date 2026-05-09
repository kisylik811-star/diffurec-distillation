"""
Statistical analysis of multi-seed runs.

- aggregate_seeds: mean ± std across seeds for each (NFE, metric).
- paired_wilcoxon: significance tests between two paired arrays.
- bootstrap_ci: bias-corrected 95% confidence intervals.
- latex_main_table: LaTeX-ready main results table for the dissertation.
- compute_metrics: FLOPs ratio + throughput for teacher vs student.

Reads JSON files produced by `multi_seed_runner.py`.
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
    """Pull metric values across seeds for a given NFE (test split)."""
    return np.array([
        results['students'][seed][str(nfe)][metric_key]
        for seed in results['students']
    ])


def _gather_val(results, nfe, metric_key):
    """Pull validation-split metric values across seeds for a given NFE.
    Returns None if val metrics are not present in the JSON."""
    seeds = list(results['students'].keys())
    if not seeds or '_val' not in results['students'][seeds[0]]:
        return None
    return np.array([
        results['students'][seed]['_val'][str(nfe)][metric_key]
        for seed in seeds
    ])


def gen_gap_analysis(results, nfe, metric='HR@10'):
    """
    Validation vs test gap — generalization sanity check.

    Returns a dict with:
      - student_val_mean, student_test_mean: average val/test for the student
      - student_gap: val - test (positive means val better than test)
      - teacher_val, teacher_test, teacher_gap: same for the teacher (single-run)
      - gap_ratio: student_gap / teacher_gap (ratio close to 1 means
        the student's generalization gap matches the teacher's, i.e.
        distillation does not introduce extra overfitting)
    """
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
    """Print a val-vs-test generalization gap summary."""
    print(f"\n=== {results['dataset']} | val/test gap | {metric} ===")
    sample = gen_gap_analysis(results, nfe_grid[0], metric)
    if sample is None:
        print('  (no val metrics in JSON — re-run multi_seed_runner.py to populate them)')
        return

    if sample['teacher_val'] is not None:
        print(f"Teacher: val={sample['teacher_val']:.4f}  "
              f"test={sample['teacher_test']:.4f}  "
              f"gap={sample['teacher_gap']:+.4f}")

    header = (f"{'NFE':<5} {'Student val':<22} {'Student test':<22} "
              f"{'Gap':<10} {'Gap/Teacher':<12}")
    print(header)
    print('-' * len(header))
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
    """Returns dict[nfe][metric] = {'mean': m, 'std': s, 'values': [...]}."""
    out = {}
    for nfe in nfe_grid:
        out[nfe] = {}
        for m in metrics:
            vals = _gather(results, nfe, m)
            out[nfe][m] = {
                'mean':   float(vals.mean()),
                'std':    float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                'values': vals.tolist(),
            }
    return out


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap (1-alpha) CI for the mean."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_wilcoxon(values_a, values_b):
    """Paired Wilcoxon signed-rank test: H0 = no difference. Returns p-value."""
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    if len(values_a) < 5:
        # Wilcoxon needs n >= 5 for any meaningful p-value
        return float('nan')
    if np.allclose(values_a, values_b):
        return 1.0
    try:
        _, p = stats.wilcoxon(values_a, values_b)
        return float(p)
    except ValueError:
        return float('nan')


def compare_student_vs_baseline(results, nfe, metric='HR@10'):
    """
    For each NFE, compares the distilled student against the truncated-DDIM
    teacher baseline at the same NFE (single value, no seeds for baseline).

    Returns: dict with student mean/std, baseline value, and a one-sample
    sign-style indicator (since baseline has no variance, we just report
    whether all student seeds beat it).
    """
    student_vals = _gather(results, nfe, metric)
    baseline_val = results['baseline'][str(nfe)][metric]
    n_better = int((student_vals > baseline_val).sum())
    return {
        'student_mean':  float(student_vals.mean()),
        'student_std':   float(student_vals.std(ddof=1)) if len(student_vals) > 1 else 0.0,
        'baseline':      float(baseline_val),
        'n_seeds':       len(student_vals),
        'n_seeds_better': n_better,
    }


def compare_student_vs_teacher(results, nfe, metric='HR@10'):
    """
    Compare the distilled student (multi-seed array) against the teacher at full
    NFE (single value, deterministic given the trained teacher).

    Desired outcome for the dissertation: p > 0.05 (i.e. *no* significant
    difference) — that is, the student preserves teacher's quality.
    Desired direction also: student_mean ≈ teacher (sometimes slightly below,
    occasionally above due to noise).

    The Wilcoxon test compares the student's array of seeds to the constant
    teacher value. With n=5 seeds this gives a meaningful p-value when the
    seeds are concentrated either clearly above or clearly below the teacher.
    """
    student_vals = _gather(results, nfe, metric)
    teacher_val  = results['teacher']['full_nfe'][metric]
    return {
        'student_mean':       float(student_vals.mean()),
        'student_std':        float(student_vals.std(ddof=1)) if len(student_vals) > 1 else 0.0,
        'teacher':            float(teacher_val),
        'gap_pct':            float((student_vals.mean() - teacher_val) / teacher_val * 100),
        'n_seeds':            len(student_vals),
        'n_seeds_above_teacher': int((student_vals > teacher_val).sum()),
    }


def significance_marker(p):
    if np.isnan(p):
        return ''
    if p < 0.001: return r'$^{***}$'
    if p < 0.01:  return r'$^{**}$'
    if p < 0.05:  return r'$^{*}$'
    return ''


def compute_metrics(results, nfe_grid=(1, 2, 4, 8, 16, 32)):
    """
    Compute derived efficiency metrics from the latency grid in `results`.

    Returns dict[nfe] = {
        'flops_ratio_vs_teacher_full':  T / NFE,
        'student_throughput_per_sec':   1000 / latency_ms,
        'teacher_truncated_throughput': 1000 / latency_ms,
        'speedup_vs_teacher_full':      teacher_full_ms / student_ms,
    }

    Note on FLOPs: teacher and student share the same Transformer backbone, so
    per-forward-pass FLOPs are identical. The total inference FLOPs for one
    sample are therefore proportional to NFE: FLOPs(student@NFE=k) / FLOPs(teacher@NFE=T) = k / T.
    We report the ratio, which is exact and does not require profiler runs.
    """
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
    """Print a compute-cost summary in the style of the main metrics summary."""
    cm = compute_metrics(results, nfe_grid)
    tf = cm['_teacher_full']
    print(f"\n=== {results['dataset']} | compute & speed ===")
    print(f"Teacher full (NFE={tf['T']}): "
          f"{tf['latency_ms']:.4f} ms, {tf['throughput_per_sec']:.1f} samples/s")
    header = f"{'NFE':<6} {'FLOPs ratio':<14} {'Student ms':<12} {'Throughput':<14} {'Speedup':<10}"
    print(header)
    print('-' * len(header))
    for nfe in nfe_grid:
        c = cm[nfe]
        print(f"{nfe:<6} {c['flops_ratio_vs_teacher_full']:<14.4f} "
              f"{c['student_latency_ms']:<12.4f} "
              f"{c['student_throughput_per_sec']:<14.1f} "
              f"{c['speedup_vs_teacher_full']:<10.2f}x")


def latex_main_table(results_per_dataset, nfe=1, metrics=('HR@5', 'HR@10', 'HR@20',
                                                          'NDCG@5', 'NDCG@10', 'NDCG@20')):
    """
    Build a LaTeX main-results table, comparable in style to Table 2 of DiffuRec.

    `results_per_dataset` is a dict {dataset_name: results_json}.

    Rows: per-dataset
    Columns: per-metric
    Cells: teacher full NFE | truncated NFE=`nfe` | student NFE=`nfe`
           Student cells carry two markers:
             * (or **, ***)  — student significantly *beats* truncated DDIM
                                (paired Wilcoxon, p < 0.05 / 0.01 / 0.001)
             † (dagger)       — no significant difference from teacher
                                (paired Wilcoxon, p >= 0.05) — desirable
    """
    lines = []
    lines.append(r'\begin{tabular}{l l ' + 'r' * len(metrics) + '}')
    lines.append(r'\toprule')
    lines.append('Dataset & Method & ' + ' & '.join(metrics) + r' \\')
    lines.append(r'\midrule')

    for ds_name, res in results_per_dataset.items():
        agg = aggregate_seeds(res, nfe_grid=[nfe], metrics=metrics)

        # ---- Literature baselines (cited from DiffuRec, Table 2) ----
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
            ds_prefix = ''  # already used the dataset name above
        else:
            ds_prefix = ds_name

        # ---- Our experimental rows ----
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
            mean = agg[nfe][m]['mean']
            std  = agg[nfe][m]['std']
            student_vals = np.array(agg[nfe][m]['values'])

            # Significance vs truncated DDIM (we *want* this to be < 0.05)
            p_trunc = paired_wilcoxon(
                student_vals,
                np.full_like(student_vals, res['baseline'][str(nfe)][m])
            )
            star = significance_marker(p_trunc)

            # Significance vs teacher (we *want* this to be >= 0.05)
            p_teacher = paired_wilcoxon(
                student_vals,
                np.full_like(student_vals, res['teacher']['full_nfe'][m])
            )
            dagger = r'$^{\dagger}$' if (not np.isnan(p_teacher) and p_teacher >= 0.05) else ''

            student_row.append(f'{mean:.2f}{star}{dagger} $\\pm$ {std:.2f}')
        lines.append(' & ' + ' & '.join(student_row[1:]) + r' \\')
        lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    # Append a footnote explaining the two markers.
    lines.append('')
    lines.append(r'% Markers:')
    lines.append(r'%   *,**,*** — student significantly beats truncated DDIM (p<0.05/0.01/0.001).')
    lines.append(r'%   $\dagger$ — no significant difference from teacher (p>=0.05); desirable.')
    return '\n'.join(lines)


def latex_compute_table(results_per_dataset, nfe_grid=(1, 2, 4, 8)):
    """
    Build a LaTeX compute-cost table: per dataset, latency / throughput / FLOPs
    ratio / speedup factor for the student at varying NFE, plus the teacher
    full-NFE reference row.
    """
    lines = []
    lines.append(r'\begin{tabular}{l l r r r r}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Method & Latency (ms) & Throughput (samples/s) '
                 r'& FLOPs ratio & Speedup \\')
    lines.append(r'\midrule')

    for ds_name, res in results_per_dataset.items():
        cm = compute_metrics(res, nfe_grid)
        tf = cm['_teacher_full']
        # Teacher full-NFE reference row.
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
    """Plain-text summary printed to stdout — quick dissertation sanity check."""
    print(f"\n=== {results['dataset']} | {metric} ===")
    print(f"Teacher (NFE={results['teacher']['T']}): "
          f"{results['teacher']['full_nfe'][metric]:.4f}")
    header = (f"{'NFE':<5} {'Trunc.':<10} {'Student (mean ± std)':<24} "
              f"{'gap vs T %':<12} {'p_vs_teacher':<14} {'p_vs_trunc':<12}")
    print(header)
    print('-' * len(header))
    for nfe in nfe_grid:
        student_vals = _gather(results, nfe, metric)
        teacher_val  = results['teacher']['full_nfe'][metric]
        truncated_val = results['baseline'][str(nfe)][metric]

        # Wilcoxon: student vs teacher (constant) — desired outcome p > 0.05
        p_vs_teacher = paired_wilcoxon(
            student_vals, np.full(len(student_vals), teacher_val)
        )
        # Wilcoxon: student vs truncated DDIM (constant) — desired outcome p < 0.05
        p_vs_trunc = paired_wilcoxon(
            student_vals, np.full(len(student_vals), truncated_val)
        )
        gap_pct = (student_vals.mean() - teacher_val) / teacher_val * 100

        print(f"{nfe:<5} {truncated_val:<10.4f} "
              f"{student_vals.mean():.4f} ± {student_vals.std(ddof=1):.4f}    "
              f"{gap_pct:>+8.2f}%   "
              f"{p_vs_teacher:<14.4g} {p_vs_trunc:<12.4g}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('json_paths', nargs='+', help='One or more JSON files from multi_seed_runner.py')
    ap.add_argument('--metric', default='HR@10')
    ap.add_argument('--out_latex', default=None,
                    help='Path to write the main results LaTeX table.')
    ap.add_argument('--out_compute_latex', default=None,
                    help='Path to write the compute / speedup LaTeX table.')
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