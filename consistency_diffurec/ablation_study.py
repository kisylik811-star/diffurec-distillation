"""
Convenience driver for the ablation studies described in the dissertation.

Two blocks are supported:

  Block 1 — RACD components (the contribution).
            Baseline = vanilla_cd. Variants:
              vanilla_cd, ndcg_only, margin_only, full_racd

  Block 2 — Design choices (justifying configuration of full_racd).
            Baseline = full_racd. Variants:
              full_racd, with_heun, with_eps, no_ema
            Each Block-2 variant flips ONE design knob from full_racd:
              with_heun  -> solver=heun (was ddim)
              with_eps   -> parametrization=eps (was xstart)
              no_ema     -> use_ema=False (was True)

Usage:
    PYTHONPATH=../DiffuRec/src python ablation_runner.py \
        --block 1 \
        --dataset amazon_beauty \
        --teacher_ckpt checkpoints/teacher_amazon_beauty.pt \
        --seeds 1997 42 2024 7 13 \
        --out_json results/ablation_block1_beauty.json

    PYTHONPATH=../DiffuRec/src python ablation_runner.py \
        --block 2 \
        --dataset amazon_beauty \
        --teacher_ckpt checkpoints/teacher_amazon_beauty.pt \
        --seeds 1997 42 2024 \
        --out_json results/ablation_block2_beauty.json

Then derive tables / figures with:

    python statistics.py results/ablation_block1_beauty.json \
        --ablation_baseline vanilla_cd \
        --ablation_variants vanilla_cd ndcg_only margin_only full_racd \
        --out_ablation_latex tables/ablation_block1.tex

    python statistics.py results/ablation_block2_beauty.json \
        --ablation_baseline full_racd \
        --ablation_variants full_racd with_heun with_eps no_ema \
        --out_ablation_latex tables/ablation_block2.tex
"""
import argparse
import subprocess
import sys
from pathlib import Path


BLOCK_VARIANTS = {
    1: ['vanilla_cd', 'ndcg_only', 'margin_only', 'full_racd'],
    2: ['full_racd', 'with_heun', 'with_eps', 'no_ema'],
}

BLOCK_BASELINE = {
    1: 'vanilla_cd',
    2: 'full_racd',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--block', type=int, choices=[1, 2], required=True,
                   help='Which ablation block: 1 = RACD components, '
                        '2 = design choices.')
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
    baseline = BLOCK_BASELINE[args.block]

    print(f'[ablation_runner] Block {args.block}')
    print(f'  variants: {variants}')
    print(f'  baseline (for paired-Wilcoxon): {baseline}')

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

    if rc == 0:
        print(f'\n[ablation_runner] done. Next steps:')
        print(f'  python statistics.py {args.out_json} \\')
        print(f'    --ablation_baseline {baseline} \\')
        print(f'    --ablation_variants {" ".join(variants)} \\')
        print(f'    --out_ablation_latex tables/ablation_block{args.block}.tex')
    sys.exit(rc)


if __name__ == '__main__':
    main()