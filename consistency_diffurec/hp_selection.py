"""
Hyperparameter selection for RCCD on Toys.

Grid search over (β, τ, lr) with selection seeds disjoint from evaluation seeds.
Best configuration chosen by mean val HR@10 (NFE=1) across selection seeds.

Selection seeds:  {1994, 2024, 91}
Evaluation seeds: {1907, 1977, 2015, 23, 88}  (used later in multi_seed_runner.py)

Output:
  - {DRIVE_BASE}/artifacts/hp_selection/{dataset}/sweep_results.json
  - Per-config artifacts under {DRIVE_BASE}/artifacts/{dataset}/{run_name}/
  - Final best config printed to stdout

Usage (from consistency_diffurec/):
  python hp_selection.py \\
      --dataset amazon_toys \\
      --selection_seeds 1994 2024 91 \\
      --betas 0.1 0.5 1.0 2.0 \\
      --taus 0.05 0.1 0.2 0.5 \\
      --lrs 1e-4 3e-4 1e-3
"""
import argparse
import json
import os
import pickle
import random
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import Data_Train, Data_Val, Data_Test
from model import create_model_diffu, Att_Diffuse_model
from trainer import model_train

from consistency_diffurec import ConsistencyStudent
from distill_trainer import distill_train, evaluate_at_nfe

DRIVE_BASE = '/content/drive/MyDrive/diffurec-distillation-results/consistency-diffurec-after-sweep'
ARTIFACTS_ROOT = f'{DRIVE_BASE}/artifacts'
LOGS_ROOT = f'{DRIVE_BASE}/logs'


def parse_args():
    p = argparse.ArgumentParser()

    # ----- Standard DiffuRec args -----
    p.add_argument('--dataset', default='amazon_toys')
    p.add_argument('--data_root', default='../datasets/data')
    p.add_argument('--max_len', type=int, default=50)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_gpu', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--emb_dropout', type=float, default=0.3)
    p.add_argument('--hidden_act', default='gelu')
    p.add_argument('--num_blocks', type=int, default=4)
    p.add_argument('--decay_step', type=int, default=100)
    p.add_argument('--gamma', type=float, default=0.1)
    p.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20])
    p.add_argument('--optimizer', default='Adam')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--loss_lambda', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=0)
    p.add_argument('--momentum', type=float, default=None)
    p.add_argument('--schedule_sampler_name', default='lossaware')
    p.add_argument('--diffusion_steps', type=int, default=32)
    p.add_argument('--lambda_uncertainty', type=float, default=0.001)
    p.add_argument('--noise_schedule', default='trunc_lin')
    p.add_argument('--rescale_timesteps', default=True)
    p.add_argument('--eval_interval', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--description', default='hp_selection')
    p.add_argument('--log_file', default=LOGS_ROOT)
    p.add_argument('--long_head', default=False)
    p.add_argument('--diversity_measure', default=False)
    p.add_argument('--epoch_time_avg', default=False)

    # ----- Teacher -----
    p.add_argument('--teacher_epochs', type=int, default=200)
    p.add_argument('--teacher_seed', type=int, default=1907)
    p.add_argument('--teacher_ckpt', default=None)
    p.add_argument('--save_teacher_ckpt',
                   default=f'{ARTIFACTS_ROOT}/{{dataset}}/teacher/teacher.pt')

    # ----- Distillation defaults -----
    p.add_argument('--distill_epochs', type=int, default=200)
    p.add_argument('--distill_eval_interval', type=int, default=5)
    p.add_argument('--distill_patience', type=int, default=10)
    p.add_argument('--cons_weight', type=float, default=1.0)
    p.add_argument('--ce_weight', type=float, default=1.0)
    p.add_argument('--ema_decay', type=float, default=0.95)

    # ----- Selection grid -----
    p.add_argument('--selection_seeds', type=int, nargs='+',
                   default=[1994, 2024, 91])
    p.add_argument('--betas', type=float, nargs='+',
                   default=[0.1, 0.5, 1.0, 2.0])
    p.add_argument('--taus', type=float, nargs='+',
                   default=[0.05, 0.1, 0.2, 0.5])
    p.add_argument('--lrs', type=float, nargs='+',
                   default=[1e-4, 3e-4, 1e-3])
    p.add_argument('--two_stage', action='store_true', default=True,
                   help='Two-stage grid: first (β, τ) at default lr, then lr at best (β, τ). '
                        'Reduces full grid (β×τ×lr) to (β×τ + lr).')
    p.add_argument('--default_lr', type=float, default=3e-4,
                   help='lr used in stage-1 of two_stage grid (frozen).')

    p.add_argument('--out_json', default=None)

    args = p.parse_args()
    args.epochs = args.teacher_epochs
    args.save_teacher_ckpt = args.save_teacher_ckpt.format(dataset=args.dataset)
    if args.out_json is None:
        args.out_json = f'{ARTIFACTS_ROOT}/hp_selection/{args.dataset}/sweep_results.json'
    return args


def fix_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    cudnn.deterministic = True
    cudnn.benchmark = False


class _DummyLogger:
    def info(self, *a, **k): pass


def load_data(args):
    path = os.path.join(args.data_root, args.dataset, 'dataset.pkl')
    with open(path, 'rb') as f:
        data_raw = pickle.load(f)
    args.item_num = len(data_raw['smap'])
    tra = Data_Train(data_raw['train'], args).get_pytorch_dataloaders()
    val = Data_Val(data_raw['train'], data_raw['val'], args).get_pytorch_dataloaders()
    tst = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'],
                    args).get_pytorch_dataloaders()
    return tra, val, tst


def train_one_config(args, teacher, beta, tau, lr, seed, logger):
    """Train one (β, τ, lr, seed). Return best val HR@10 at NFE=1."""
    args.random_seed = seed
    args.contrast_weight = beta
    args.contrast_temperature = tau
    args.distill_lr = lr

    fix_seed(seed)
    tra, val, tst = load_data(args)
    device = torch.device(args.device)
    student = ConsistencyStudent(teacher, args, ema_decay=args.ema_decay).to(device)

    run_name = f'hpsel_seed{seed}_beta{beta}_tau{tau}_lr{lr}'
    log_dir = os.path.join(LOGS_ROOT, args.dataset, 'hp_selection')
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f'{run_name}.csv')

    best_student = distill_train(student, teacher.diffu, tra, val, tst, args,
                                 logger, log_csv_path=csv_path, run_name=run_name)
    val_metrics = evaluate_at_nfe(best_student, val, num_steps=1, device=device)
    return val_metrics['HR@10'], run_name


def main():
    args = parse_args()
    Path(os.path.dirname(args.out_json) or '.').mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.save_teacher_ckpt) or '.').mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    logger = _DummyLogger()

    print(f'\n=== HP selection on {args.dataset} ===')
    print(f'  Selection seeds: {args.selection_seeds}')
    print(f'  β grid:  {args.betas}')
    print(f'  τ grid:  {args.taus}')
    print(f'  lr grid: {args.lrs}')
    print(f'  Strategy: {"two-stage" if args.two_stage else "full"}')

    # --- Teacher (single seed; defines weights for all student trainings) ---
    fix_seed(args.teacher_seed)
    tra, val, tst = load_data(args)
    print(f'[Data] item_num={args.item_num}')

    teacher_ckpt = args.teacher_ckpt or args.save_teacher_ckpt
    teacher = Att_Diffuse_model(create_model_diffu(args), args).to(device)
    if os.path.exists(teacher_ckpt):
        print(f'[Teacher] loading from {teacher_ckpt}')
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    else:
        print(f'[Teacher] training from scratch (seed={args.teacher_seed})')
        teacher, _ = model_train(tra, val, tst, teacher, args, logger)
        torch.save(teacher.state_dict(), args.save_teacher_ckpt)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- Build grid ---
    if args.two_stage:
        # Stage 1: vary (β, τ) at fixed default_lr
        configs_stage1 = [(b, t, args.default_lr) for b, t in product(args.betas, args.taus)]
        configs = configs_stage1
    else:
        configs = list(product(args.betas, args.taus, args.lrs))

    print(f'\n[Stage 1] configs: {len(configs)}, '
          f'total runs: {len(configs) * len(args.selection_seeds)}')

    grid_results = {}  # key: (β, τ, lr) → {seed: val_hr10}
    t0 = time.time()
    for ci, (beta, tau, lr) in enumerate(configs):
        seed_vals = {}
        for seed in args.selection_seeds:
            print(f'\n  [{ci+1}/{len(configs)}] β={beta}, τ={tau}, lr={lr}, '
                  f'seed={seed}  elapsed={time.time()-t0:.0f}s')
            v, run_name = train_one_config(args, teacher, beta, tau, lr, seed, logger)
            seed_vals[seed] = v
            print(f'    val HR@10 = {v:.4f}')
        grid_results[(beta, tau, lr)] = seed_vals

    # --- Pick best by mean val HR@10 ---
    config_means = {cfg: float(np.mean(list(seed_vals.values())))
                    for cfg, seed_vals in grid_results.items()}
    best_cfg = max(config_means, key=config_means.get)
    best_beta, best_tau, best_lr = best_cfg

    print(f'\n[Stage 1 best] β={best_beta}, τ={best_tau}, lr={best_lr} '
          f'→ mean val HR@10 = {config_means[best_cfg]:.4f}')

    # --- Stage 2: vary lr around best β, τ ---
    if args.two_stage:
        remaining_lrs = [lr for lr in args.lrs if lr != best_lr]
        if remaining_lrs:
            print(f'\n[Stage 2] varying lr ∈ {remaining_lrs} at β={best_beta}, τ={best_tau}')
            for lr in remaining_lrs:
                seed_vals = {}
                for seed in args.selection_seeds:
                    print(f'\n  β={best_beta}, τ={best_tau}, lr={lr}, seed={seed}')
                    v, _ = train_one_config(args, teacher, best_beta, best_tau, lr, seed, logger)
                    seed_vals[seed] = v
                    print(f'    val HR@10 = {v:.4f}')
                grid_results[(best_beta, best_tau, lr)] = seed_vals
                config_means[(best_beta, best_tau, lr)] = float(np.mean(list(seed_vals.values())))
            # Recompute best across lrs (β, τ fixed)
            best_lr = max([lr for lr in args.lrs],
                          key=lambda lr: config_means.get((best_beta, best_tau, lr), -1))
            best_cfg = (best_beta, best_tau, best_lr)

    print(f'\n========== FINAL BEST CONFIG ==========')
    print(f'  β  = {best_cfg[0]}')
    print(f'  τ  = {best_cfg[1]}')
    print(f'  lr = {best_cfg[2]}')
    print(f'  mean val HR@10 = {config_means[best_cfg]:.4f}')
    print(f'  per-seed: {grid_results[best_cfg]}')

    # --- Serialize ---
    out = {
        'dataset': args.dataset,
        'selection_seeds': args.selection_seeds,
        'grid': {
            f'beta={k[0]}_tau={k[1]}_lr={k[2]}': {
                'beta': k[0], 'tau': k[1], 'lr': k[2],
                'per_seed': v, 'mean_val_HR10': config_means[k],
            }
            for k, v in grid_results.items()
        },
        'best': {
            'beta': best_cfg[0], 'tau': best_cfg[1], 'lr': best_cfg[2],
            'mean_val_HR10': config_means[best_cfg],
            'per_seed': grid_results[best_cfg],
        },
    }
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'\n[Save] {args.out_json}')


if __name__ == '__main__':
    main()