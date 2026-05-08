"""
Convenience driver for the ablation studies described in the dissertation.

The ablation protocol has two blocks:

  Block 1 — RACD components (the contribution itself).
            Baseline = vanilla_cd. Variants:
              vanilla_cd, ndcg_only, margin_only, full_racd
            Each variant is compared against vanilla_cd by paired Wilcoxon.

  Block 2 — Design choices (justifying configuration, not contribution).
            Baseline = full_racd. Variants:
              full_racd, with_ddim, with_eps, no_ema
            Each variant is compared against full_racd by paired Wilcoxon.

Block 2 requires variant flags (--solver, --parametrization, --use_ema) to be
honoured by the student / trainer. Until those are wired up, only Block 1
will run cleanly; Block 2 will train but the design knobs will fall back
to defaults (a warning is printed at startup so this is obvious).

Usage:
    PYTHONPATH=../DiffuRec/src python ablation_runner.py \
        --dataset amazon_beauty \
        --teacher_ckpt checkpoints/teacher_amazon_beauty.pt \
        --block 1 \
        --seeds 1997 42 2024 7 13 \
        --out_json results/ablation_block1_beauty.json

After running, hand the resulting JSON to:

    python statistics.py results/ablation_block1_beauty.json \
        --ablation_baseline vanilla_cd \
        --ablation_variants vanilla_cd ndcg_only margin_only full_racd \
        --out_ablation_latex tables/ablation_block1.tex

    python plots.py --json_paths results/ablation_block1_beauty.json ...
"""
import argparse
import subprocess
import sys
from pathlib import Path


BLOCK_VARIANTS = {
    1: ['vanilla_cd', 'ndcg_only', 'margin_only', 'full_racd'],
    2: ['full_racd', 'with_ddim', 'with_eps', 'no_ema'],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--block', type=int, choices=[1, 2], required=True,
                   help='Which ablation block: 1 = RACD components, '
                        '2 = design choices.')
    # All other args are forwarded verbatim to multi_seed_runner.
    p.add_argument('--dataset', required=True)
    p.add_argument('--teacher_ckpt', default=None)
    p.add_argument('--data_root', default='../datasets/data')
    p.add_argument('--seeds', type=int, nargs='+', default=[1997, 42, 2024])
    p.add_argument('--nfe_grid', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--distill_epochs', type=int, default=100)
    p.add_argument('--distill_patience', type=int, default=10)
    p.add_argument('--out_json', required=True)
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                   help='Anything after --extra is forwarded to multi_seed_runner.')
    return p.parse_args()


def main():
    args = parse_args()
    variants = BLOCK_VARIANTS[args.block]
    if args.block == 2:
        print('[ablation_runner] WARNING: Block 2 variants depend on solver / '
              'parametrization / EMA switches in the student. Confirm the code '
              'honours those overrides before relying on these results.')

    Path(Path(args.out_json).parent or '.').mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, 'multi_seed_runner.py',
        '--dataset', args.dataset,
        '--data_root', args.data_root,
        '--seeds', *map(str, args.seeds),
        '--nfe_grid', *map(str, args.nfe_grid),
        '--variants', *variants,
        '--distill_epochs', str(args.distill_epochs),
        '--distill_patience', str(args.distill_patience),
        '--out_json', args.out_json,
    ]
    if args.teacher_ckpt:
        cmd += ['--teacher_ckpt', args.teacher_ckpt]
    cmd += list(args.extra)

    print('[ablation_runner] running:', ' '.join(cmd))
    rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == '__main__':
    main()