"""
Statistical analysis of multi-seed runs.

Key concepts in this module:
  * paired_wilcoxon          : two paired samples (variant_A vs variant_B with the
                               same seeds). Used for ablation: e.g. RACD vs Vanilla CD.
  * one_sample_wilcoxon_vs   : one sample vs a deterministic constant. Used for
                               main-results: e.g. RACD-seeds vs Truncated-DDIM
                               (single number, no seeds).
  * bootstrap_ci             : percentile bootstrap (1 - alpha) CI for the mean.
  * ablation_pairwise_table  : LaTeX table comparing each variant against a chosen
                               baseline variant via paired Wilcoxon. Used for
                               Block 1 (vs Vanilla CD) and Block 2 (vs Full RACD).

Reads JSON files produced by `multi_seed_runner.py`. The JSON layout supports
multiple variants per seed:

    results['variants'][variant_name][seed][nfe][metric] -> float

Backwards compat: if a JSON has the legacy `students` key, we treat it as a
single variant called 'student'.
"""
import json
import numpy as np
from scipy import stats


# --------------------------------------------------------------------- #
#  IO and shape helpers                                                 #
# --------------------------------------------------------------------- #
def load_results(path):
    with open(path) as f:
        return json.load(f)


def variants_in(results):
    """List of variant names available in this results JSON."""
    if 'variants' in results:
        return list(results['variants'].keys())
    if 'students' in results:
        return ['student']  # legacy
    return []


def _seed_dict(results, variant):
    """variant -> {seed: {nfe: {metric: float}}}."""
    if 'variants' in results:
        return results['variants'][variant]
    # legacy: only 'students'
    return results['students']


def _gather(results, variant, nfe, metric_key):
    """Pull metric values across seeds for a (variant, NFE)."""
    sd = _seed_dict(results, variant)
    return np.array([sd[seed][str(nfe)][metric_key] for seed in sd])


# --------------------------------------------------------------------- #
#  Aggregation                                                          #
# --------------------------------------------------------------------- #
def aggregate_seeds(results, variant='racd', nfe_grid=(1, 2, 4, 8),
                    metrics=('HR@5', 'HR@10', 'HR@20',
                             'NDCG@5', 'NDCG@10', 'NDCG@20')):
    """For one variant: dict[nfe][metric] = {'mean','std','values'}."""
    out = {}
    for nfe in nfe_grid:
        out[nfe] = {}
        for m in metrics:
            vals = _gather(results, variant, nfe, m)
            out[nfe][m] = {
                'mean':   float(vals.mean()),
                'std':    float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                'values': vals.tolist(),
            }
    return out


# --------------------------------------------------------------------- #
#  CIs and significance tests                                           #
# --------------------------------------------------------------------- #
def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap (1 - alpha) CI for the mean."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_wilcoxon(values_a, values_b):
    """
    Paired Wilcoxon signed-rank test.

    Use when both `values_a` and `values_b` are seed-aligned samples of the
    same length (e.g. variant_A on seed_i vs variant_B on seed_i for the
    same i). Returns p-value (two-sided).

    Wilcoxon needs n >= 5 to be meaningful; we return NaN below that.
    """
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    if len(a) != len(b):
        raise ValueError(f'paired Wilcoxon requires equal lengths, got {len(a)} vs {len(b)}')
    if len(a) < 5:
        return float('nan')
    if np.allclose(a, b):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b)
        return float(p)
    except ValueError:
        return float('nan')


def one_sample_wilcoxon_vs(values, constant):
    """
    One-sample Wilcoxon signed-rank test against a deterministic constant.

    Use when `values` are seed samples of one variant and `constant` is a
    single number with no seed variability (e.g. Truncated DDIM, which is
    deterministic at fixed teacher). H0: median(values - constant) = 0.

    This is the correct alternative to a paired test against an artificial
    array `[constant] * len(values)` — that path technically runs but
    inflates the test's effective n and is harder to defend.
    """
    v = np.asarray(values, dtype=float)
    if len(v) < 5:
        return float('nan')
    diffs = v - float(constant)
    if np.allclose(diffs, 0):
        return 1.0
    try:
        _, p = stats.wilcoxon(diffs)
        return float(p)
    except ValueError:
        return float('nan')


def significance_marker(p):
    if np.isnan(p):
        return ''
    if p < 0.001: return r'$^{***}$'
    if p < 0.01:  return r'$^{**}$'
    if p < 0.05:  return r'$^{*}$'
    return ''


# --------------------------------------------------------------------- #
#  Public comparison helpers                                            #
# --------------------------------------------------------------------- #
def compare_variants_paired(results, variant_a, variant_b, nfe, metric='HR@10'):
    """
    Paired comparison between two variants on the same seeds.
    Used for ablation: e.g. ('full_racd', 'vanilla_cd').
    """
    a = _gather(results, variant_a, nfe, metric)
    b = _gather(results, variant_b, nfe, metric)
    p = paired_wilcoxon(a, b)
    return {
        'variant_a': variant_a,
        'variant_b': variant_b,
        'mean_a':    float(a.mean()),
        'std_a':     float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        'mean_b':    float(b.mean()),
        'std_b':     float(b.std(ddof=1)) if len(b) > 1 else 0.0,
        'delta':     float(a.mean() - b.mean()),
        'p_wilcoxon': p,
    }


def compare_variant_vs_baseline(results, variant, nfe, metric='HR@10'):
    """
    Compare a seeded variant against a deterministic baseline (Truncated DDIM).
    Uses one-sample Wilcoxon and reports n_seeds_better as a robust addendum.
    """
    student_vals = _gather(results, variant, nfe, metric)
    baseline_val = results['baseline'][str(nfe)][metric]
    p = one_sample_wilcoxon_vs(student_vals, baseline_val)
    n_better = int((student_vals > baseline_val).sum())
    return {
        'variant':       variant,
        'student_mean':  float(student_vals.mean()),
        'student_std':   float(student_vals.std(ddof=1)) if len(student_vals) > 1 else 0.0,
        'baseline':      float(baseline_val),
        'n_seeds':       len(student_vals),
        'n_seeds_better': n_better,
        'p_wilcoxon_one_sample': p,
    }


# --------------------------------------------------------------------- #
#  Compute / latency derived metrics                                    #
# --------------------------------------------------------------------- #
def compute_metrics(results, nfe_grid=(1, 2, 4, 8, 16, 32), variant='racd'):
    """
    Derived efficiency metrics from the latency grid.

    `variant` selects which student's latency to use (variants are trained
    independently but share architecture, so latency is essentially identical;
    we still pick the one explicitly named).
    """
    T = results['teacher']['T']
    teacher_full_ms = results['latency']['teacher_full']

    # `latency.student` may be variant-keyed or flat (legacy)
    student_lat_root = results['latency']['student']
    if variant in student_lat_root:
        student_lat = student_lat_root[variant]
    else:
        student_lat = student_lat_root  # legacy flat

    out = {}
    for nfe in nfe_grid:
        nfe_str = str(nfe)
        s_lat = student_lat[nfe_str]
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


def compute_metrics_text(results, nfe_grid=(1, 2, 4, 8, 16, 32), variant='racd'):
    cm = compute_metrics(results, nfe_grid, variant=variant)
    tf = cm['_teacher_full']
    print(f"\n=== {results['dataset']} | compute & speed ({variant}) ===")
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


# --------------------------------------------------------------------- #
#  LaTeX tables                                                         #
# --------------------------------------------------------------------- #
def latex_main_table(results_per_dataset, variant='racd', nfe=1,
                     metrics=('HR@5', 'HR@10', 'HR@20',
                              'NDCG@5', 'NDCG@10', 'NDCG@20')):
    """
    Main results table: rows are datasets x methods, columns are metrics.

    Methods per dataset:
      - Teacher (full NFE)               : single number
      - Truncated DDIM (NFE=`nfe`)       : single number
      - <variant> (NFE=`nfe`)            : mean ± std with one-sample Wilcoxon
                                            vs Truncated DDIM
    """
    lines = []
    lines.append(r'\begin{tabular}{l l ' + 'r' * len(metrics) + '}')
    lines.append(r'\toprule')
    lines.append('Dataset & Method & ' + ' & '.join(metrics) + r' \\')
    lines.append(r'\midrule')

    for ds_name, res in results_per_dataset.items():
        agg = aggregate_seeds(res, variant=variant, nfe_grid=[nfe], metrics=metrics)

        # Teacher full
        teacher_cells = [f'{res["teacher"]["full_nfe"][m]:.2f}' for m in metrics]
        lines.append(f'{ds_name} & Teacher (NFE={res["teacher"]["T"]}) & '
                     + ' & '.join(teacher_cells) + r' \\')

        # Truncated DDIM
        baseline_cells = [f'{res["baseline"][str(nfe)][m]:.2f}' for m in metrics]
        lines.append(f' & Truncated DDIM (NFE={nfe}) & '
                     + ' & '.join(baseline_cells) + r' \\')

        # Variant (RACD or whichever requested)
        var_cells = []
        for m in metrics:
            mean = agg[nfe][m]['mean']
            std  = agg[nfe][m]['std']
            p    = one_sample_wilcoxon_vs(agg[nfe][m]['values'],
                                          res['baseline'][str(nfe)][m])
            var_cells.append(f'{mean:.2f}{significance_marker(p)} $\\pm$ {std:.2f}')
        lines.append(f' & {variant.upper()} (NFE={nfe}) & '
                     + ' & '.join(var_cells) + r' \\')
        lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)


def ablation_pairwise_table(results, variants, baseline_variant, nfe=1,
                            metrics=('HR@10', 'NDCG@10'), title=None):
    """
    Generic ablation table: each row is a variant, compared (paired Wilcoxon)
    against `baseline_variant`. The baseline row itself is shown without a
    p-value for reference.

    Use cases:
      Block 1 (RACD components):  baseline_variant='vanilla_cd',
                                  variants=['vanilla_cd','+ndcg','+margin','full_racd']
      Block 2 (design choices):   baseline_variant='full_racd',
                                  variants=['full_racd','with_ddim','with_eps','no_ema']
    """
    lines = []
    if title:
        lines.append(f'% {title}')
    n_metric_cols = len(metrics)
    lines.append(r'\begin{tabular}{l ' + 'r ' * n_metric_cols + r'r}')
    lines.append(r'\toprule')
    header = ['Variant'] + list(metrics) + [f'p vs {baseline_variant}']
    lines.append(' & '.join(header) + r' \\')
    lines.append(r'\midrule')

    base_vals = {m: _gather(results, baseline_variant, nfe, m) for m in metrics}

    for v in variants:
        cells = [v]
        for m in metrics:
            vals = _gather(results, v, nfe, m)
            mean = vals.mean()
            std  = vals.std(ddof=1) if len(vals) > 1 else 0.0
            cells.append(f'{mean:.2f} $\\pm$ {std:.2f}')

        if v == baseline_variant:
            cells.append('---')
        else:
            # Use the first metric for the headline p-value (standard practice)
            head_m = metrics[0]
            vals_v = _gather(results, v, nfe, head_m)
            p = paired_wilcoxon(vals_v, base_vals[head_m])
            cells.append(f'{p:.3g}{significance_marker(p)}' if not np.isnan(p) else 'n/a')

        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)


def latex_compute_table(results_per_dataset, variant='racd', nfe_grid=(1, 2, 4, 8)):
    lines = []
    lines.append(r'\begin{tabular}{l l r r r r}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Method & Latency (ms) & Throughput (samples/s) '
                 r'& FLOPs ratio & Speedup \\')
    lines.append(r'\midrule')

    for ds_name, res in results_per_dataset.items():
        cm = compute_metrics(res, nfe_grid, variant=variant)
        tf = cm['_teacher_full']
        lines.append(f'{ds_name} & Teacher (NFE={tf["T"]}) '
                     f'& {tf["latency_ms"]:.3f} '
                     f'& {tf["throughput_per_sec"]:.1f} '
                     f'& {tf["flops_ratio"]:.3f} '
                     f'& 1.00$\\times$ \\\\')
        for nfe in nfe_grid:
            c = cm[nfe]
            lines.append(f' & {variant.upper()} (NFE={nfe}) '
                         f'& {c["student_latency_ms"]:.3f} '
                         f'& {c["student_throughput_per_sec"]:.1f} '
                         f'& {c["flops_ratio_vs_teacher_full"]:.3f} '
                         f'& {c["speedup_vs_teacher_full"]:.2f}$\\times$ \\\\')
        lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)


# --------------------------------------------------------------------- #
#  Plain-text summary                                                   #
# --------------------------------------------------------------------- #
def summary_table_text(results, variant='racd', nfe_grid=(1, 2, 4, 8),
                       metric='HR@10'):
    print(f"\n=== {results['dataset']} | {variant} | {metric} ===")
    print(f"Teacher (NFE={results['teacher']['T']}): {results['teacher']['full_nfe'][metric]:.4f}")
    print(f"{'NFE':<6} {'Truncated':<12} {variant + ' (mean ± std)':<28} {'p (one-sample W)':<16}")
    for nfe in nfe_grid:
        cmp_ = compare_variant_vs_baseline(results, variant, nfe, metric)
        print(f"{nfe:<6} {cmp_['baseline']:<12.4f} "
              f"{cmp_['student_mean']:.4f} ± {cmp_['student_std']:.4f}      "
              f"{cmp_['p_wilcoxon_one_sample']:<16.4g}")


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('json_paths', nargs='+',
                    help='One or more JSON files from multi_seed_runner.py')
    ap.add_argument('--variant', default='racd',
                    help='Which student variant is the headline result.')
    ap.add_argument('--metric', default='HR@10')
    ap.add_argument('--out_latex', default=None,
                    help='Path to write the main results LaTeX table.')
    ap.add_argument('--out_compute_latex', default=None,
                    help='Path to write the compute / speedup LaTeX table.')
    ap.add_argument('--ablation_baseline', default=None,
                    help='If set, dump an ablation table with this baseline '
                         '(e.g. vanilla_cd or full_racd).')
    ap.add_argument('--ablation_variants', nargs='*', default=None,
                    help='Variants to include in the ablation table.')
    ap.add_argument('--out_ablation_latex', default=None)
    ap.add_argument('--nfe_grid', type=int, nargs='+', default=[1, 2, 4, 8])
    args = ap.parse_args()

    per_dataset = {}
    for path in args.json_paths:
        r = load_results(path)
        per_dataset[r['dataset']] = r
        summary_table_text(r, variant=args.variant, nfe_grid=args.nfe_grid,
                           metric=args.metric)
        compute_metrics_text(r, nfe_grid=args.nfe_grid, variant=args.variant)

    if args.out_latex:
        with open(args.out_latex, 'w') as f:
            f.write(latex_main_table(per_dataset, variant=args.variant))
        print(f'\nMain results LaTeX table -> {args.out_latex}')

    if args.out_compute_latex:
        with open(args.out_compute_latex, 'w') as f:
            f.write(latex_compute_table(per_dataset, variant=args.variant,
                                        nfe_grid=tuple(args.nfe_grid)))
        print(f'Compute LaTeX table -> {args.out_compute_latex}')

    if args.ablation_baseline and args.ablation_variants:
        # Use the first JSON's results as the ablation source (one dataset per ablation
        # is the standard convention).
        first = list(per_dataset.values())[0]
        tex = ablation_pairwise_table(first,
                                      variants=args.ablation_variants,
                                      baseline_variant=args.ablation_baseline,
                                      nfe=args.nfe_grid[0],
                                      metrics=('HR@10', 'NDCG@10'))
        if args.out_ablation_latex:
            with open(args.out_ablation_latex, 'w') as f:
                f.write(tex)
            print(f'Ablation LaTeX table -> {args.out_ablation_latex}')
        else:
            print('\n' + tex)