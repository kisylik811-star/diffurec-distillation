"""
Stand-alone latency table builder.

Measures wall-clock inference latency over 1000 forward passes at batch_size=1
with NFE=1 (RCCD) vs NFE=T (DiffuRec teacher), on seed 1907.

Output (inline, no PDF/PNG):

  Dataset    Method      Steps  Latency (ms)  Speedup vs DiffuRec  HR@10
  toys       DiffuRec    32     60.0123       1.00x                X.XXX
             RCCD        1       1.8745       32.02x               Y.YYY
  beauty     DiffuRec    32     ...
             ...

Datasets and student checkpoints are loaded from
{DRIVE_BASE}/artifacts/{dataset}/{run_name}/student_final.pt.

By default uses the seed=1907 RCCD checkpoint with the production (β, τ).
"""
import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import Data_Test
from model import create_model_diffu, Att_Diffuse_model
from consistency_diffurec import ConsistencyStudent
from distill_trainer import evaluate_at_nfe, evaluate_teacher_full_nfe

DRIVE_BASE = '/content/drive/MyDrive/diffurec-distillation-results/consistency-diffurec-after-sweep'
ARTIFACTS_ROOT = f'{DRIVE_BASE}/artifacts'
LATENCY_SEED = 1907


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+',
                   default=['amazon_toys', 'amazon_beauty', 'ml-1m'])
    p.add_argument('--data_root', default='../datasets/data')
    p.add_argument('--device', default='cuda')

    # ----- DiffuRec architecture (mirror teacher config) -----
    p.add_argument('--max_len', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=1,
                   help='Per-query latency = batch_size=1.')
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--emb_dropout', type=float, default=0.3)
    p.add_argument('--hidden_act', default='gelu')
    p.add_argument('--num_blocks', type=int, default=4)
    p.add_argument('--diffusion_steps', type=int, default=32)
    p.add_argument('--lambda_uncertainty', type=float, default=0.001)
    p.add_argument('--noise_schedule', default='trunc_lin')
    p.add_argument('--rescale_timesteps', default=True)
    p.add_argument('--schedule_sampler_name', default='lossaware')

    # ----- Production hyperparameters (selected on Toys) -----
    p.add_argument('--seed', type=int, default=LATENCY_SEED,
                   help='Seed for selecting the student checkpoint. '
                        'Latency itself is invariant to seed.')
    p.add_argument('--beta', type=float, default=2.0)
    p.add_argument('--tau', type=float, default=0.1)
    p.add_argument('--nfe_student', type=int, default=1)

    # ----- Measurement -----
    p.add_argument('--n_runs', type=int, default=1000)
    p.add_argument('--n_warmup', type=int, default=100)

    p.add_argument('--artifacts_root', default=ARTIFACTS_ROOT)
    p.add_argument('--out_json', default=None)

    args = p.parse_args()
    if args.out_json is None:
        args.out_json = f'{ARTIFACTS_ROOT}/latency_table.json'
    return args


def fix_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    cudnn.deterministic = True
    cudnn.benchmark = False


def load_data_single(args, dataset):
    args.dataset = dataset
    path = os.path.join(args.data_root, dataset, 'dataset.pkl')
    with open(path, 'rb') as f:
        data_raw = pickle.load(f)
    args.item_num = len(data_raw['smap'])
    loader = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'],
                       args).get_pytorch_dataloaders()
    return loader


@torch.no_grad()
def time_forward(fn, n_warmup, n_runs, device):
    for _ in range(n_warmup):
        _ = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        _ = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0 / n_runs


def main():
    args = parse_args()
    Path(os.path.dirname(args.out_json) or '.').mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    rows = []
    for dataset in args.datasets:
        print(f'\n=== {dataset} ===')
        fix_seed(LATENCY_SEED)
        loader = load_data_single(args, dataset)

        teacher_ckpt = f'{args.artifacts_root}/{dataset}/teacher/teacher.pt'
        student_run = f'seed{args.seed}_beta{args.beta}_tau{args.tau}'
        student_ckpt = f'{args.artifacts_root}/{dataset}/{student_run}/student_final.pt'

        if not os.path.exists(teacher_ckpt):
            print(f'  [skip] missing teacher: {teacher_ckpt}')
            continue
        if not os.path.exists(student_ckpt):
            print(f'  [skip] missing student: {student_ckpt}')
            continue

        teacher = Att_Diffuse_model(create_model_diffu(args), args).to(device)
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        teacher.eval()

        student = ConsistencyStudent(teacher, args).to(device)
        student.load_state_dict(torch.load(student_ckpt, map_location=device))
        student.eval()

        # Quality (HR@10): from cached evaluation, recomputed for self-containedness
        teacher_metrics = evaluate_teacher_full_nfe(teacher, loader, device)
        student_metrics = evaluate_at_nfe(student, loader, num_steps=args.nfe_student,
                                          device=device)

        # Latency: single query, n_runs=1000
        seq, target = next(iter(loader))
        seq, target = seq.to(device), target.to(device)

        def _teacher_fwd():
            _, rep, *_ = teacher(seq, target, train_flag=False)
            return teacher.diffu_rep_pre(rep)

        def _student_fwd():
            return student.predict_scores(seq, num_steps=args.nfe_student)

        t_lat = time_forward(_teacher_fwd, args.n_warmup, args.n_runs, device)
        s_lat = time_forward(_student_fwd, args.n_warmup, args.n_runs, device)
        # Normalize to per-sample (batch_size may be >1 if test loader doesn't honor bs=1)
        t_lat_per = t_lat / seq.size(0)
        s_lat_per = s_lat / seq.size(0)

        speedup_t = 1.0
        speedup_s = t_lat_per / s_lat_per if s_lat_per > 0 else float('inf')

        rows.append({
            'dataset':       dataset,
            'teacher_steps': args.diffusion_steps,
            'teacher_ms':    t_lat_per,
            'teacher_HR10':  teacher_metrics['HR@10'],
            'student_steps': args.nfe_student,
            'student_ms':    s_lat_per,
            'student_HR10':  student_metrics['HR@10'],
            'speedup':       speedup_s,
            'batch_size':    seq.size(0),
        })
        print(f'  Teacher (NFE={args.diffusion_steps}): '
              f'{t_lat_per:.4f} ms/query  HR@10={teacher_metrics["HR@10"]:.4f}')
        print(f'  RCCD    (NFE={args.nfe_student}): '
              f'{s_lat_per:.4f} ms/query  HR@10={student_metrics["HR@10"]:.4f}')
        print(f'  Speedup: {speedup_s:.2f}x')

    # --- Print final inline table ---
    print('\n' + '=' * 80)
    print(f'LATENCY TABLE  (n_runs={args.n_runs}, batch_size={args.batch_size}, '
          f'seed={LATENCY_SEED}, NFE_student={args.nfe_student})')
    print('=' * 80)
    hdr = (f"{'Dataset':<14} {'Method':<14} {'Steps':<7} "
           f"{'Latency (ms)':<14} {'Speedup vs DiffuRec':<22} {'HR@10':<8}")
    print(hdr); print('-' * len(hdr))
    for r in rows:
        print(f"{r['dataset']:<14} {'DiffuRec':<14} {r['teacher_steps']:<7} "
              f"{r['teacher_ms']:<14.4f} {'1.00x':<22} {r['teacher_HR10']:<8.4f}")
        speed_str = f"{r['speedup']:.2f}x"
        print(f"{'':<14} {'RCCD (ours)':<14} {r['student_steps']:<7} "
              f"{r['student_ms']:<14.4f} {speed_str:<22} {r['student_HR10']:<8.4f}")
        print('-' * len(hdr))

    with open(args.out_json, 'w') as f:
        json.dump({
            'n_runs': args.n_runs, 'batch_size': args.batch_size,
            'seed': LATENCY_SEED, 'nfe_student': args.nfe_student,
            'rows': rows,
        }, f, indent=2, default=float)
    print(f'\n[Save] {args.out_json}')


if __name__ == '__main__':
    main()