"""
Multi-seed evaluation runner for consistency distillation.

Runs the teacher ONCE (cached), then for each of 5 evaluation seeds trains:
  - RCCD student   (full architecture with InfoNCE contrastive loss)
  - CD-only student (β=0, ablation baseline)

Evaluates each at NFE ∈ {1, 2, 4, 8} → inference-steps ablation comes for free.
Measures 1000-inference latency at NFE=1, batch_size=1, seed=1907 (single
measurement; latency is hardware-dependent and seed-invariant).

NOTE: Teacher is trained on a single seed (no variance available).
      Reported student-vs-teacher comparisons should use one-sample Wilcoxon
      as a conservative test, with effect size reported alongside.

Default seeds {1907, 1977, 2015, 23, 88} are *evaluation* seeds; for
hyperparameter selection use {1994, 2024, 91} in hp_selection.py.

All artifacts under DRIVE_BASE = /content/drive/MyDrive/...
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

from utils import Data_Train, Data_Val, Data_Test
from model import create_model_diffu, Att_Diffuse_model
from trainer import model_train

from consistency_diffurec import ConsistencyStudent
from distill_trainer import (
    distill_train, evaluate_at_nfe, evaluate_teacher_full_nfe,
)
from evaluation import evaluate_teacher_truncated, measure_latency_grid

DRIVE_BASE = '/content/drive/MyDrive/consistency_diffurec/multi_seed_runs'
ARTIFACTS_ROOT = f'{DRIVE_BASE}/artifacts'
LOGS_ROOT = f'{DRIVE_BASE}/logs'
RESULTS_ROOT = f'{DRIVE_BASE}/results'
# Учительские чекпоинты лежат отдельно — общие для всех экспериментов.
TEACHERS_ROOT = '/content/drive/MyDrive/consistency_diffurec/teachers_ckpts'

# Seed for the single-shot latency measurement (deterministic init only;
# latency itself is invariant to seed).
LATENCY_SEED = 1907


def parse_args():
    p = argparse.ArgumentParser()

    # ----- DiffuRec args -----
    p.add_argument('--dataset', default='amazon_beauty')
    p.add_argument('--data_root', default='../datasets/data')
    p.add_argument('--max_len', type=int, default=50)
    p.add_argument('--device', default='cuda')
    p.add_argument('--only_latency', action='store_true',
               help='Skip all training; load existing checkpoints + JSON, '
                    'remeasure latency only, update JSON in place.')
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
    p.add_argument('--log_file', default=LOGS_ROOT)
    p.add_argument('--long_head', default=False)
    p.add_argument('--diversity_measure', default=False)
    p.add_argument('--epoch_time_avg', default=False)

    # ----- Teacher -----
    p.add_argument('--teacher_epochs', type=int, default=200)
    p.add_argument('--teacher_ckpt', default=None,
                   help='Путь к предобученному учителю. По умолчанию: '
                        f'{TEACHERS_ROOT}/teacher_{{dataset}}.pt')
    p.add_argument('--save_teacher_ckpt',
                   default=f'{TEACHERS_ROOT}/teacher_{{dataset}}.pt',
                   help='Куда сохранять учителя если придётся обучать с нуля. '
                        'В норме не используется.')

    # ----- Distillation hyperparameters (selected on Toys via hp_selection.py) -----
    p.add_argument('--distill_lr', type=float, default=1e-3,
                   help='Student learning rate (selected on Toys grid).')
    p.add_argument('--contrast_weight', type=float, default=2.0,
                   help='InfoNCE loss weight β (selected on Toys grid).')
    p.add_argument('--contrast_temperature', type=float, default=0.1,
                   help='InfoNCE temperature τ (selected on Toys grid).')
    p.add_argument('--distill_epochs', type=int, default=200)
    p.add_argument('--distill_eval_interval', type=int, default=5)
    p.add_argument('--distill_patience', type=int, default=10)
    p.add_argument('--cons_weight', type=float, default=1.0)
    p.add_argument('--ce_weight', type=float, default=1.0)
    p.add_argument('--ema_decay', type=float, default=0.95)

    # ----- Multi-seed evaluation -----
    p.add_argument('--seeds', type=int, nargs='+',
                   default=[1907, 1977, 2015, 23, 88],
                   help='Evaluation seeds. Disjoint from selection seeds '
                        '{1994, 2024, 91} used by hp_selection.py.')
    p.add_argument('--teacher_seed', type=int, default=1907,
                   help='Seed used for teacher training. Single value — '
                        'teacher has no reported variance.')
    p.add_argument('--nfe_grid', type=int, nargs='+', default=[1, 2, 4, 8],
                   help='NFE values evaluated for inference-steps ablation.')
    p.add_argument('--run_ablation', action='store_true', default=True,
                   help='Run CD-only baseline (β=0) alongside RCCD for ablation.')
    p.add_argument('--no_ablation', dest='run_ablation', action='store_false')
    p.add_argument('--out_json', default=None,
                   help='Output JSON path. Default: '
                        '{DRIVE_BASE}/artifacts/results/multiseed_{dataset}.json')

    # ----- 1000-inference latency table -----
    p.add_argument('--latency_n_runs', type=int, default=1000)
    p.add_argument('--latency_batch_size', type=int, default=1)

    args = p.parse_args()
    args.epochs = args.teacher_epochs
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


def _build_latency_loader(args, target_batch_size):
    """Build a separate test loader with batch_size=1 for accurate per-query latency."""
    saved_bs = args.batch_size
    args.batch_size = target_batch_size
    path = os.path.join(args.data_root, args.dataset, 'dataset.pkl')
    with open(path, 'rb') as f:
        data_raw = pickle.load(f)
    args.item_num = len(data_raw['smap'])
    loader = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'],
                       args).get_pytorch_dataloaders()
    args.batch_size = saved_bs
    return loader


@torch.no_grad()
def measure_single_query_latency(model, loader, device, n_warmup=100, n_runs=1000,
                                 mode='student', num_steps=1):
    """Per-query (batch_size=1) latency, averaged over n_runs forward passes.

    Single-shot measurement; latency is hardware-dependent and seed-invariant,
    so we do not aggregate across evaluation seeds.
    """
    seq, target = next(iter(loader))
    seq, target = seq.to(device), target.to(device)
    model.eval()

    def _run_once():
        if mode == 'student':
            return model.predict_scores(seq, num_steps=num_steps)
        _, rep, *_ = model(seq, target, train_flag=False)
        return model.diffu_rep_pre(rep)

    for _ in range(n_warmup):
        _ = _run_once()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        _ = _run_once()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_ms = (time.time() - t0) * 1000.0
    return total_ms / n_runs / seq.size(0)

def _rerun_latency_only(args):
    """Load existing JSON + checkpoints, remeasure latency, save back.

    Re-uses: teacher.pt and seed{LATENCY_SEED}_beta{β}_tau{τ}/student_final.pt
    """
    device = torch.device(args.device)

    # Load existing JSON
    if not os.path.exists(args.out_json):
        raise SystemExit(f'--only_latency requires existing JSON at {args.out_json}')
    with open(args.out_json) as f:
        results = json.load(f)
    print(f'[only_latency] loaded existing results from {args.out_json}')

    # Load data (need for latency batch sampling)
    fix_seed(args.teacher_seed)
    _, _, test_loader = load_data(args)

    # Load teacher
    teacher_ckpt = args.teacher_ckpt or args.save_teacher_ckpt
    teacher = Att_Diffuse_model(create_model_diffu(args), args).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Load student (seed=1907 RCCD)
    run_name = f'seed{LATENCY_SEED}_beta{args.contrast_weight}_tau{args.contrast_temperature}'
    student_ckpt = f'{ARTIFACTS_ROOT}/{args.dataset}/{run_name}/student_final.pt'
    if not os.path.exists(student_ckpt):
        raise SystemExit(f'Missing student checkpoint: {student_ckpt}')
    student = ConsistencyStudent(teacher, args, ema_decay=args.ema_decay).to(device)
    student.load_state_dict(torch.load(student_ckpt, map_location=device))
    student.eval()
    print(f'[only_latency] loaded teacher + student (run={run_name})')

    # Measure
    print('\n[Latency] grid measurement')
    sample_batch = next(iter(test_loader))
    results['latency'] = measure_latency_grid(
        teacher, student, sample_batch, device, args.nfe_grid,
    )

    print(f'[Latency] single-query (bs={args.latency_batch_size}, '
          f'n_runs={args.latency_n_runs}, seed={LATENCY_SEED}, NFE=1)')
    fix_seed(LATENCY_SEED)
    bs1_loader = _build_latency_loader(args, target_batch_size=args.latency_batch_size)
    t_lat = measure_single_query_latency(
        teacher, bs1_loader, device,
        n_runs=args.latency_n_runs, mode='teacher')
    s_lat = measure_single_query_latency(
        student, bs1_loader, device,
        n_runs=args.latency_n_runs, mode='student', num_steps=1)
    results['latency_single_query'] = {
        'batch_size': args.latency_batch_size,
        'n_runs':     args.latency_n_runs,
        'seed':       LATENCY_SEED,
        'teacher_ms':       t_lat,
        'student_ms_nfe1':  s_lat,
        'speedup':          t_lat / s_lat if s_lat > 0 else float('inf'),
    }
    print(f'  Teacher: {t_lat:.4f} ms, Student NFE=1: {s_lat:.4f} ms, '
          f'speedup={results["latency_single_query"]["speedup"]:.2f}x')

    # Save back
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\n[Done] latency fields updated in {args.out_json}')

def main():
    args = parse_args()
    args.save_teacher_ckpt = args.save_teacher_ckpt.format(dataset=args.dataset)
    if args.out_json is None:
        args.out_json = f'{RESULTS_ROOT}/multiseed_{args.dataset}.json'

    if args.only_latency:
        return _rerun_latency_only(args)

    Path(os.path.dirname(args.out_json) or '.').mkdir(parents=True, exist_ok=True)
    # NOTE: папку для save_teacher_ckpt создаём только если реально обучаем учителя.

    device = torch.device(args.device)
    logger = _DummyLogger()

    print(f'[Config] β={args.contrast_weight}, τ={args.contrast_temperature}, '
          f'lr={args.distill_lr}')
    print(f'[Config] eval seeds: {args.seeds}')
    print(f'[Config] ablation (CD-only β=0): {"YES" if args.run_ablation else "NO"}')
    print(f'[Config] output: {args.out_json}')

    # --- Load data once at teacher seed ---
    fix_seed(args.teacher_seed)
    tra_loader, val_loader, test_loader = load_data(args)
    print(f'[Data] {args.dataset}: item_num={args.item_num}')

    results = {
        'dataset': args.dataset,
        'config': {k: v for k, v in vars(args).items()
                   if isinstance(v, (int, float, str, bool, list, type(None)))},
        'teacher': {},
        'students': {},            # keyed by seed → RCCD
        'students_baseline': {},   # keyed by seed → CD-only (β=0)
        'baseline': {},            # truncated teacher (NFE varying)
        'baseline_val': {},
        'latency': {},
    }

    # --- Teacher ---
    teacher_ckpt = args.teacher_ckpt or args.save_teacher_ckpt
    teacher = Att_Diffuse_model(create_model_diffu(args), args).to(device)
    if os.path.exists(teacher_ckpt):
        print(f'[Teacher] loading from {teacher_ckpt}')
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    else:
        print(f'[Teacher] training from scratch ({args.teacher_epochs} epochs, '
              f'seed={args.teacher_seed})')
        # Создаём папку под чекпоинт только перед обучением.
        Path(os.path.dirname(args.save_teacher_ckpt) or '.').mkdir(
            parents=True, exist_ok=True)
        teacher, _ = model_train(tra_loader, val_loader, test_loader, teacher, args, logger)
        torch.save(teacher.state_dict(), args.save_teacher_ckpt)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print('[Teacher] full-NFE eval')
    results['teacher']['full_nfe']     = evaluate_teacher_full_nfe(teacher, test_loader, device)
    results['teacher']['full_nfe_val'] = evaluate_teacher_full_nfe(teacher, val_loader, device)
    results['teacher']['T'] = args.diffusion_steps
    results['teacher']['_single_seed_note'] = (
        'Teacher trained on a single seed; reported variance is for students only. '
        'Use one-sample Wilcoxon (conservative) for student-vs-teacher tests.'
    )

    print('[Baseline] truncated DDIM at varying NFE')
    for nfe in args.nfe_grid:
        m_test = evaluate_teacher_truncated(teacher, test_loader, num_steps=nfe, device=device)
        m_val = evaluate_teacher_truncated(teacher, val_loader, num_steps=nfe, device=device)
        results['baseline'][str(nfe)] = m_test
        results['baseline_val'][str(nfe)] = m_val
        print(f'  truncated NFE={nfe}: test={m_test}')

    # --- Per-seed loop: RCCD + (optionally) CD-only ---
    target_beta, target_tau = args.contrast_weight, args.contrast_temperature
    seed_configs = [('rccd', target_beta, target_tau)]
    if args.run_ablation:
        seed_configs.append(('baseline', 0.0, target_tau))

    final_student_for_latency = None

    for seed in args.seeds:
        print(f'\n========== Seed {seed} ==========')
        for variant_name, beta, tau in seed_configs:
            args.random_seed = seed
            args.contrast_weight = beta
            args.contrast_temperature = tau

            print(f'\n  [Variant: {variant_name}] β={beta}, τ={tau}, '
                  f'lr={args.distill_lr}, seed={seed}')

            fix_seed(seed)
            tra_s, val_s, tst_s = load_data(args)
            student = ConsistencyStudent(teacher, args,
                                         ema_decay=args.ema_decay).to(device)

            run_name = (f"seed{seed}_beta{beta}_tau{tau}"
                        + ('_baseline' if variant_name == 'baseline' else ''))
            log_dir = os.path.join(LOGS_ROOT, args.dataset)
            os.makedirs(log_dir, exist_ok=True)
            csv_path = os.path.join(log_dir, f'{run_name}.csv')

            best_student = distill_train(
                student, teacher.diffu, tra_s, val_s, tst_s,
                args, logger, log_csv_path=csv_path, run_name=run_name,
            )

            seed_data = {'_run_name': run_name, '_val': {}}
            for nfe in args.nfe_grid:
                m_test = evaluate_at_nfe(best_student, tst_s, num_steps=nfe, device=device)
                m_val = evaluate_at_nfe(best_student, val_s, num_steps=nfe, device=device)
                seed_data[str(nfe)] = m_test
                seed_data['_val'][str(nfe)] = m_val
                print(f'    seed={seed} [{variant_name}] NFE={nfe}: '
                      f'test={m_test["HR@10"]:.4f}  val={m_val["HR@10"]:.4f}')

            store_key = 'students' if variant_name == 'rccd' else 'students_baseline'
            results[store_key][str(seed)] = seed_data

            if variant_name == 'rccd' and seed == LATENCY_SEED:
                final_student_for_latency = best_student

        # Save intermediate results after every seed in case of interruption
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2, default=float)

    # --- Latency: grid + 1000-inference single-query measurement on seed=1907 ---
    print('\n[Latency] grid measurement (bs=test_default)')
    sample_batch = next(iter(test_loader))
    if final_student_for_latency is None:
        # Fallback: use the last trained student
        final_student_for_latency = best_student
    results['latency'] = measure_latency_grid(
        teacher, final_student_for_latency, sample_batch, device, args.nfe_grid,
    )

    print(f'[Latency] single-query measurement '
          f'(batch_size={args.latency_batch_size}, n_runs={args.latency_n_runs}, '
          f'seed={LATENCY_SEED}, NFE=1)')
    fix_seed(LATENCY_SEED)
    bs1_loader = _build_latency_loader(args, target_batch_size=args.latency_batch_size)

    t_lat_1q = measure_single_query_latency(
        teacher, bs1_loader, device,
        n_runs=args.latency_n_runs, mode='teacher',
    )
    s_lat_1q = measure_single_query_latency(
        final_student_for_latency, bs1_loader, device,
        n_runs=args.latency_n_runs, mode='student', num_steps=1,
    )
    results['latency_single_query'] = {
        'batch_size': args.latency_batch_size,
        'n_runs':     args.latency_n_runs,
        'seed':       LATENCY_SEED,
        'teacher_ms':       t_lat_1q,
        'student_ms_nfe1':  s_lat_1q,
        'speedup':          t_lat_1q / s_lat_1q if s_lat_1q > 0 else float('inf'),
    }
    print(f'  Teacher (NFE={args.diffusion_steps}): {t_lat_1q:.4f} ms/query')
    print(f'  RCCD    (NFE=1):                      {s_lat_1q:.4f} ms/query')
    print(f'  Speedup: {results["latency_single_query"]["speedup"]:.2f}x')

    # --- Persist ---
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\n[Done] results -> {args.out_json}')


if __name__ == '__main__':
    main()