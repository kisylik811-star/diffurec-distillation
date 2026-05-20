"""
Hyperparameter transferability sanity check.

Takes the best (β, τ, lr) selected on Toys via hp_selection.py and verifies
that the Toys-optimal τ remains close to optimal on Beauty and ML-1M by
varying ONLY τ in a small local window {0.5τ*, τ*, 2τ*} on a single seed
(1994 — a selection seed, so this does not contaminate evaluation seeds).

If τ_toys ≠ argmax_τ on either dataset, that's a signal that hyperparameter
transfer is not safe and per-dataset tuning may be warranted. Otherwise the
transfer is empirically justified.

Usage:
  python sanity_check.py \\
      --datasets amazon_beauty ml-1m \\
      --best_beta 2.0 --best_tau 0.1 --best_lr 1e-3 \\
      --tau_window 0.05 0.1 0.2 \\
      --sanity_seed 1994
"""
import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import Data_Train, Data_Val, Data_Test
from model import create_model_diffu, Att_Diffuse_model
from trainer import model_train

from consistency_diffurec import ConsistencyStudent
from distill_trainer import distill_train, evaluate_at_nfe

DRIVE_BASE = '/content/drive/MyDrive/consistency_diffurec/sanity_check'
ARTIFACTS_ROOT = f'{DRIVE_BASE}/artifacts'
LOGS_ROOT = f'{DRIVE_BASE}/logs'
# Учительские чекпоинты лежат отдельно — общие для всех экспериментов.
TEACHERS_ROOT = '/content/drive/MyDrive/consistency_diffurec/teachers_ckpts'


def parse_args():
    p = argparse.ArgumentParser()

    # ----- Standard DiffuRec args -----
    p.add_argument('--datasets', nargs='+', default=['amazon_beauty', 'ml-1m'])
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
    p.add_argument('--description', default='sanity_check')
    p.add_argument('--log_file', default=LOGS_ROOT)
    p.add_argument('--long_head', default=False)
    p.add_argument('--diversity_measure', default=False)
    p.add_argument('--epoch_time_avg', default=False)

    # ----- Teacher -----
    p.add_argument('--teacher_epochs', type=int, default=200)
    p.add_argument('--teacher_seed', type=int, default=1907)
    p.add_argument('--teacher_ckpt', default=None,
                   help='Путь к предобученному учителю. По умолчанию: '
                        f'{TEACHERS_ROOT}/teacher_{{dataset}}.pt')
    p.add_argument('--save_teacher_ckpt',
                   default=f'{TEACHERS_ROOT}/teacher_{{dataset}}.pt',
                   help='Куда сохранять учителя если придётся обучать с нуля.')

    # ----- Distillation -----
    p.add_argument('--distill_epochs', type=int, default=200)
    p.add_argument('--distill_eval_interval', type=int, default=5)
    p.add_argument('--distill_patience', type=int, default=10)
    p.add_argument('--cons_weight', type=float, default=1.0)
    p.add_argument('--ce_weight', type=float, default=1.0)
    p.add_argument('--ema_decay', type=float, default=0.95)

    # ----- Sanity-check inputs (from hp_selection.py) -----
    p.add_argument('--best_beta', type=float, required=True,
                   help='Best β found on Toys via hp_selection.py')
    p.add_argument('--best_tau', type=float, required=True,
                   help='Best τ found on Toys via hp_selection.py')
    p.add_argument('--best_lr', type=float, required=True,
                   help='Best lr found on Toys via hp_selection.py')
    p.add_argument('--tau_window', type=float, nargs='+', default=None,
                   help='τ values to test. Default: {0.5τ*, τ*, 2τ*}')
    p.add_argument('--sanity_seed', type=int, default=1994,
                   help='Single seed for sanity check. Must be a SELECTION seed, '
                        'not an evaluation seed.')

    args = p.parse_args()
    args.epochs = args.teacher_epochs
    if args.tau_window is None:
        args.tau_window = [args.best_tau * 0.5, args.best_tau, args.best_tau * 2.0]
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


def run_for_dataset(args, dataset, logger):
    args.dataset = dataset
    args.save_teacher_ckpt = f'{TEACHERS_ROOT}/teacher_{dataset}.pt'
    # NOTE: папку под чекпоинт учителя создаём только если будем его обучать.
    device = torch.device(args.device)

    # --- Teacher ---
    fix_seed(args.teacher_seed)
    tra, val, tst = load_data(args)
    print(f'\n[{dataset}] item_num={args.item_num}')

    teacher = Att_Diffuse_model(create_model_diffu(args), args).to(device)
    teacher_ckpt = args.teacher_ckpt or args.save_teacher_ckpt
    if os.path.exists(teacher_ckpt):
        print(f'[{dataset}] [Teacher] loading from {teacher_ckpt}')
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    else:
        print(f'[{dataset}] [Teacher] training from scratch (seed={args.teacher_seed})')
        Path(os.path.dirname(args.save_teacher_ckpt) or '.').mkdir(
            parents=True, exist_ok=True)
        teacher, _ = model_train(tra, val, tst, teacher, args, logger)
        torch.save(teacher.state_dict(), args.save_teacher_ckpt)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- Sanity check: vary τ at fixed β, lr ---
    results = {}
    for tau in args.tau_window:
        print(f'\n[{dataset}] β={args.best_beta}, τ={tau}, lr={args.best_lr}, '
              f'seed={args.sanity_seed}')
        args.random_seed = args.sanity_seed
        args.contrast_weight = args.best_beta
        args.contrast_temperature = tau
        args.distill_lr = args.best_lr
        fix_seed(args.sanity_seed)
        tra_s, val_s, tst_s = load_data(args)
        student = ConsistencyStudent(teacher, args, ema_decay=args.ema_decay).to(device)
        run_name = (f'sanity_{dataset}_seed{args.sanity_seed}_'
                    f'beta{args.best_beta}_tau{tau}_lr{args.best_lr}')
        log_dir = os.path.join(LOGS_ROOT, dataset, 'sanity')
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f'{run_name}.csv')
        best_student = distill_train(student, teacher.diffu, tra_s, val_s, tst_s,
                                     args, logger, log_csv_path=csv_path,
                                     run_name=run_name)
        val_metrics = evaluate_at_nfe(best_student, val_s, num_steps=1, device=device)
        results[tau] = val_metrics['HR@10']
        print(f'  τ={tau} → val HR@10 = {results[tau]:.4f}')

    best_tau_here = max(results, key=results.get)
    transfers_ok = abs(best_tau_here - args.best_tau) < 1e-9
    return {
        'tau_results': results,
        'best_tau_local': best_tau_here,
        'toys_optimal_tau': args.best_tau,
        'transfers_cleanly': bool(transfers_ok),
        'relative_gap_pct': float(
            (results[args.best_tau] - max(results.values()))
            / max(results.values()) * 100
        ),
    }


def main():
    args = parse_args()
    logger = _DummyLogger()

    print(f'=== Sanity check ===')
    print(f'  Toys-optimal: β={args.best_beta}, τ={args.best_tau}, lr={args.best_lr}')
    print(f'  τ window: {args.tau_window}')
    print(f'  sanity seed: {args.sanity_seed} (selection seed; not in eval pool)')
    print(f'  datasets: {args.datasets}')

    summary = {
        'toys_optimal': {
            'beta': args.best_beta, 'tau': args.best_tau, 'lr': args.best_lr,
        },
        'sanity_seed': args.sanity_seed,
        'tau_window': args.tau_window,
        'datasets': {},
    }

    for ds in args.datasets:
        summary['datasets'][ds] = run_for_dataset(args, ds, logger)

    # Report
    print('\n========== SUMMARY ==========')
    for ds, info in summary['datasets'].items():
        marker = '✓ transfers' if info['transfers_cleanly'] else '✗ DOES NOT transfer'
        print(f'\n[{ds}] {marker}')
        print(f'  Toys-optimal τ = {info["toys_optimal_tau"]}')
        print(f'  Local-optimal τ = {info["best_tau_local"]}')
        print(f'  Gap from local optimum at Toys-τ: {info["relative_gap_pct"]:.2f}%')
        for tau, hr in info['tau_results'].items():
            star = ' ★' if tau == info['best_tau_local'] else ''
            print(f'    τ={tau}: val HR@10 = {hr:.4f}{star}')


if __name__ == '__main__':
    main()