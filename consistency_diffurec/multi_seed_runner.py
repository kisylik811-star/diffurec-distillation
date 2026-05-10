"""
Multi-seed experiment runner for the consistency distillation pipeline.

Runs the teacher once (cached on disk), then trains students with different
random seeds, evaluating each at NFE = 1, 2, 4, 8. Also evaluates the teacher
with truncated DDIM (the "naive truncation" baseline) at the same NFEs.

All results are written to a single JSON for downstream statistics + plots.

Usage (from `consistency_diffurec/`):
        --dataset amazon_beauty \
        --data_root ../datasets/data \
        --teacher_epochs 200 \
        --distill_epochs 200 \
        --seeds 1997 42 2024 7 13 \
        --out_json results/beauty_multiseed.json
"""
import argparse
import copy
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import Data_Train, Data_Val, Data_Test
from model import create_model_diffu, Att_Diffuse_model
from trainer import model_train

from consistency_diffurec import ConsistencyStudent
from distill_trainer import distill_train, evaluate_at_nfe, evaluate_teacher_full_nfe
from evaluation import evaluate_teacher_truncated, measure_latency_grid


def parse_args():
    p = argparse.ArgumentParser()
    # DiffuRec args (subset that matters)
    p.add_argument('--dataset', default='amazon_beauty')
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
    p.add_argument('--description', default='multiseed')
    p.add_argument('--log_file', default='log/')
    p.add_argument('--long_head', default=False)
    p.add_argument('--diversity_measure', default=False)
    p.add_argument('--epoch_time_avg', default=False)

    # Teacher
    p.add_argument('--teacher_epochs', type=int, default=200)
    p.add_argument('--teacher_ckpt', default=None,
                   help='Reuse a pre-trained teacher checkpoint if provided.')
    p.add_argument('--save_teacher_ckpt', default='checkpoints/teacher_{dataset}.pt')

    # Distillation
    p.add_argument('--distill_lr', type=float, default=1e-3)
    p.add_argument('--distill_epochs', type=int, default=200)
    p.add_argument('--distill_eval_interval', type=int, default=5)
    p.add_argument('--distill_patience', type=int, default=10)
    p.add_argument('--cons_weight', type=float, default=1.0)
    p.add_argument('--ce_weight', type=float, default=1.0)
    p.add_argument('--ema_decay', type=float, default=0.95)

    # Multi-seed
    p.add_argument('--seeds', type=int, nargs='+', default=[1997, 42, 2024, 7, 13])
    p.add_argument('--teacher_seed', type=int, default=1997)
    p.add_argument('--nfe_grid', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    p.add_argument('--out_json', default='results/multiseed.json')

    # Required by Att_Diffuse_model but not parsed in main: filled later
    args = p.parse_args()
    args.epochs = args.teacher_epochs  # used by trainer for teacher
    return args


def fix_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    cudnn.deterministic = True
    cudnn.benchmark = False


def load_data(args):
    path = os.path.join(args.data_root, args.dataset, 'dataset.pkl')
    with open(path, 'rb') as f:
        data_raw = pickle.load(f)
    args.item_num = len(data_raw['smap'])
    tra = Data_Train(data_raw['train'], args).get_pytorch_dataloaders()
    val = Data_Val(data_raw['train'], data_raw['val'], args).get_pytorch_dataloaders()
    tst = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args).get_pytorch_dataloaders()
    return tra, val, tst


class _DummyLogger:
    def info(self, *a, **k): pass


def main():
    args = parse_args()
    args.save_teacher_ckpt = args.save_teacher_ckpt.format(dataset=args.dataset)
    Path(os.path.dirname(args.out_json) or '.').mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.save_teacher_ckpt) or '.').mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger = _DummyLogger()

    # --- Load data once ---
    fix_seed(args.teacher_seed)
    tra_loader, val_loader, test_loader = load_data(args)
    print(f'[Data] {args.dataset}: item_num={args.item_num}')

    results = {
        'dataset':  args.dataset,
        'config':   {k: v for k, v in vars(args).items()
                     if isinstance(v, (int, float, str, bool, list))},
        'teacher':  {},
        'students': {},     # keyed by seed
        'baseline': {},     # truncated teacher at varying NFE
        'latency':  {},
    }

    # --- Teacher ---
    teacher_ckpt = args.teacher_ckpt or args.save_teacher_ckpt
    teacher = Att_Diffuse_model(create_model_diffu(args), args).to(device)
    if os.path.exists(teacher_ckpt):
        print(f'[Teacher] loading from {teacher_ckpt}')
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    else:
        print(f'[Teacher] training from scratch ({args.teacher_epochs} epochs)')
        teacher, _ = model_train(tra_loader, val_loader, test_loader, teacher, args, logger)
        torch.save(teacher.state_dict(), args.save_teacher_ckpt)
        print(f'[Teacher] saved to {args.save_teacher_ckpt}')

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print('[Teacher] evaluating at full NFE on test and val')
    results['teacher']['full_nfe']     = evaluate_teacher_full_nfe(teacher, test_loader, device)
    results['teacher']['full_nfe_val'] = evaluate_teacher_full_nfe(teacher, val_loader, device)
    results['teacher']['T'] = args.diffusion_steps

    # --- Baseline: teacher with truncated DDIM (test and val) ---
    print('[Baseline] truncated DDIM at varying NFE (test and val)')
    results['baseline_val'] = {}
    for nfe in args.nfe_grid:
        m_test = evaluate_teacher_truncated(teacher, test_loader, num_steps=nfe, device=device)
        m_val  = evaluate_teacher_truncated(teacher, val_loader,  num_steps=nfe, device=device)
        results['baseline'][str(nfe)]     = m_test
        results['baseline_val'][str(nfe)] = m_val
        print(f'  truncated NFE={nfe}: test={m_test}  val={m_val}')

    # --- Multi-seed students ---
    for seed in args.seeds:
        print(f'\n========== Seed {seed} ==========')
        fix_seed(seed)

        # Reload data with this seed (reshuffles training order via DataLoader)
        tra_s, val_s, tst_s = load_data(args)

        student = ConsistencyStudent(teacher, args, ema_decay=args.ema_decay).to(device)

        # Per-seed CSV path. Lives in `logs/<dataset>/seed_<seed>.csv` so that
        # plots.py can load any/all of them later.
        log_dir = os.path.join('logs', args.dataset)
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f'seed_{seed}.csv')

        best_student = distill_train(
            student, teacher.diffu, tra_s, val_s, tst_s,
            args, logger,
            log_csv_path=csv_path,
        )

        seed_results = {'_log_csv': csv_path, '_val': {}}
        for nfe in args.nfe_grid:
            m_test = evaluate_at_nfe(best_student, tst_s, num_steps=nfe, device=device)
            m_val  = evaluate_at_nfe(best_student, val_s, num_steps=nfe, device=device)
            seed_results[str(nfe)]         = m_test
            seed_results['_val'][str(nfe)] = m_val
            print(f'  Student seed={seed} NFE={nfe}: test={m_test}  val={m_val}')
        results['students'][str(seed)] = seed_results

    # --- Latency grid (one batch is enough) ---
    print('\n[Latency] measuring on first test batch')
    sample_batch = next(iter(test_loader))
    final_student = best_student  # last trained student is fine for latency
    results['latency'] = measure_latency_grid(
        teacher, final_student, sample_batch, device, args.nfe_grid
    )

    # --- Persist ---
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\n[Done] results -> {args.out_json}')


if __name__ == '__main__':
    main()