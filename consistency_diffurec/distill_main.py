"""
Entry point for consistency distillation of DiffuRec.

Three modes:
  1. Single run: one (beta, tau) configuration with one seed.
  2. Sweep:     grid over (beta, tau), single seed. For hyperparameter selection.
  3. Multi-seed: same (beta, tau), multiple seeds. For final reported numbers.

Always saves:
  - artifacts/<dataset>/teacher/teacher.pt        (copy of teacher checkpoint)
  - artifacts/<dataset>/teacher/reference.json    (teacher metrics & latency)
  - artifacts/<dataset>/<run_name>/{...}          (per-run artifacts)

For multi-seed mode, also writes a consolidated multi-seed JSON for
downstream multi-seed statistics in analyze.py.
"""
import argparse
import json
import logging
import os
import pickle
import random
import shutil
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
    distill_train,
    evaluate_at_nfe,
    evaluate_teacher_full_nfe,
    measure_inference_latency,
)
from evaluation import evaluate_teacher_truncated, measure_latency_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='amazon_beauty')
    p.add_argument('--log_file', default='log/')
    p.add_argument('--random_seed', type=int, default=1997)
    p.add_argument('--max_len', type=int, default=50)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--num_gpu', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--emb_dropout', type=float, default=0.3)
    p.add_argument('--hidden_act', default='gelu')
    p.add_argument('--num_blocks', type=int, default=4)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--decay_step', type=int, default=100)
    p.add_argument('--gamma', type=float, default=0.1)
    p.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20])
    p.add_argument('--optimizer', type=str, default='Adam')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--loss_lambda', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=0)
    p.add_argument('--momentum', type=float, default=None)
    p.add_argument('--schedule_sampler_name', type=str, default='lossaware')
    p.add_argument('--diffusion_steps', type=int, default=32)
    p.add_argument('--lambda_uncertainty', type=float, default=0.001)
    p.add_argument('--noise_schedule', default='trunc_lin')
    p.add_argument('--rescale_timesteps', default=True)
    p.add_argument('--eval_interval', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--description', type=str, default='RCCD')
    p.add_argument('--long_head', default=False)
    p.add_argument('--diversity_measure', default=False)
    p.add_argument('--epoch_time_avg', default=False)

    p.add_argument('--teacher_ckpt', type=str, default=None)
    p.add_argument('--save_teacher_ckpt', type=str, default='checkpoints/teacher.pt')
    p.add_argument('--distill_lr', type=float, default=1e-3)
    p.add_argument('--distill_epochs', type=int, default=200)
    p.add_argument('--distill_eval_interval', type=int, default=5)
    p.add_argument('--distill_patience', type=int, default=10)
    p.add_argument('--cons_weight', type=float, default=1.0)
    p.add_argument('--ce_weight', type=float, default=1.0)
    p.add_argument('--ema_decay', type=float, default=0.95)

    p.add_argument('--contrast_weight', type=float, default=0.5)
    p.add_argument('--contrast_temperature', type=float, default=0.1)

    # ----- Mode selection -----
    p.add_argument('--sweep', action='store_true',
                   help='Run a grid sweep over (beta, tau) with single seed.')
    p.add_argument('--sweep_betas', type=float, nargs='+',
                   default=[0.1, 0.5, 1.0, 2.0])
    p.add_argument('--sweep_taus', type=float, nargs='+',
                   default=[0.05, 0.1, 0.2])
    p.add_argument('--include_baseline', action='store_true',
                   help='In sweep mode, also include beta=0 (CD baseline).')

    p.add_argument('--multiseed', action='store_true',
                   help='Run the SAME (beta, tau) with multiple seeds. '
                        'Use after sweep to get final reported numbers.')
    p.add_argument('--seeds', type=int, nargs='+', default=[1997, 42, 2024])
    p.add_argument('--multiseed_baseline', action='store_true',
                   help='In multiseed mode, also run beta=0 baseline for each seed.')
    p.add_argument('--nfe_grid', type=int, nargs='+', default=[1, 2, 4, 8],
                   help='NFE values evaluated in multi-seed mode and latency grid.')
    p.add_argument('--out_multiseed_json', type=str, default=None,
                   help='Path for consolidated multi-seed JSON (analyze.py reads this).')

    p.add_argument('--data_root', type=str, default='../datasets/data')

    return p.parse_args()


def fix_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    cudnn.deterministic = True
    cudnn.benchmark = False


def setup_logging(args, suffix=''):
    """One log file per invocation, not per configuration."""
    log_dir = os.path.join(args.log_file, args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    fname = os.path.join(log_dir, f'distill_{suffix}{stamp}.log' if suffix
                                  else f'distill_{stamp}.log')
    # Remove old handlers (matters if called more than once in one Python session)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, filename=fname,
                        format='%(asctime)s - %(message)s', filemode='w')
    return logging.getLogger(__name__)


def load_data(args):
    """Load dataset once. Sets args.item_num as side effect."""
    path = os.path.join(args.data_root, args.dataset, 'dataset.pkl')
    with open(path, 'rb') as f:
        data_raw = pickle.load(f)
    args.item_num = len(data_raw['smap'])
    tra = Data_Train(data_raw['train'], args).get_pytorch_dataloaders()
    val = Data_Val(data_raw['train'], data_raw['val'], args).get_pytorch_dataloaders()
    tst = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'],
                    args).get_pytorch_dataloaders()
    return tra, val, tst, data_raw


def load_or_train_teacher(args, tra_loader, val_loader, test_loader, logger):
    """Load teacher checkpoint or train from scratch. Always copies the final
    teacher checkpoint to artifacts/<dataset>/teacher/teacher.pt."""
    device = torch.device(args.device)
    teacher = Att_Diffuse_model(create_model_diffu(args), args).to(device)

    if args.teacher_ckpt and os.path.exists(args.teacher_ckpt):
        print(f'Loading teacher from {args.teacher_ckpt}')
        logger.info(f'Loading teacher from {args.teacher_ckpt}')
        teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=args.device))
        teacher_src = args.teacher_ckpt
    else:
        print('Training teacher from scratch...')
        logger.info('Training teacher from scratch...')
        teacher, _ = model_train(tra_loader, val_loader, test_loader, teacher, args, logger)
        os.makedirs(os.path.dirname(args.save_teacher_ckpt) or '.', exist_ok=True)
        torch.save(teacher.state_dict(), args.save_teacher_ckpt)
        teacher_src = args.save_teacher_ckpt
        print(f'Teacher saved to {args.save_teacher_ckpt}')

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Teacher reference metrics
    print('\n=== Teacher (NFE=T) Test Performance ===')
    logger.info('=== Teacher (NFE=T) Test Performance ===')
    teacher_metrics_test = evaluate_teacher_full_nfe(teacher, test_loader, device)
    teacher_metrics_val  = evaluate_teacher_full_nfe(teacher, val_loader, device)
    print(teacher_metrics_test)
    logger.info(teacher_metrics_test)

    sample_batch = next(iter(test_loader))
    teacher_ms = measure_inference_latency(teacher, sample_batch, device, mode='teacher')
    print(f'Teacher latency (NFE=T): {teacher_ms:.4f} ms/sample')
    logger.info(f'Teacher latency (NFE=T): {teacher_ms:.4f} ms/sample')

    # Persist teacher reference to standard location
    teacher_ref_dir = Path('artifacts') / args.dataset / 'teacher'
    teacher_ref_dir.mkdir(parents=True, exist_ok=True)

    teacher_ckpt_copy = teacher_ref_dir / 'teacher.pt'
    if not teacher_ckpt_copy.exists() and os.path.exists(teacher_src):
        shutil.copy(teacher_src, teacher_ckpt_copy)
        print(f'[Save] teacher checkpoint copy -> {teacher_ckpt_copy}')

    # Save teacher config alongside (so analyze.py can recover architecture params)
    teacher_config = {k: (v if isinstance(v, (int, float, str, bool, list, type(None)))
                          else str(v))
                      for k, v in vars(args).items()}
    with open(teacher_ref_dir / 'config.json', 'w') as f:
        json.dump(teacher_config, f, indent=2)

    with open(teacher_ref_dir / 'reference.json', 'w') as f:
        json.dump({
            'metrics_full_nfe_test': teacher_metrics_test,
            'metrics_full_nfe_val':  teacher_metrics_val,
            'latency_ms':            teacher_ms,
            'T':                     args.diffusion_steps,
        }, f, indent=2)

    return teacher, teacher_metrics_test, teacher_metrics_val, teacher_ms


def run_single_configuration(args, teacher, tra_loader, val_loader, test_loader,
                             contrast_weight, contrast_temperature, logger,
                             run_name=None):
    """Train one (beta, tau, seed) configuration. Artifacts go to disk."""
    args.contrast_weight = contrast_weight
    args.contrast_temperature = contrast_temperature

    print(f'\n{"="*70}')
    print(f'Configuration: beta={contrast_weight}, tau={contrast_temperature}, '
          f'seed={args.random_seed}')
    print(f'{"="*70}')
    logger.info(f'Config: beta={contrast_weight}, tau={contrast_temperature}, '
                f'seed={args.random_seed}')

    fix_seed(args.random_seed)
    student = ConsistencyStudent(teacher, args, ema_decay=args.ema_decay)

    if run_name is None:
        run_name = (f"seed{args.random_seed}_beta{contrast_weight}_"
                    f"tau{contrast_temperature}")

    log_dir = os.path.join('logs', args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f'{run_name}.csv')

    best_student = distill_train(student, teacher.diffu,
                                 tra_loader, val_loader, test_loader,
                                 args, logger,
                                 log_csv_path=csv_path,
                                 run_name=run_name)
    return best_student, run_name


def run_sweep(args, teacher, tra_loader, val_loader, test_loader, logger):
    configs = []
    if args.include_baseline:
        configs.append((0.0, 0.1))
    for beta in args.sweep_betas:
        for tau in args.sweep_taus:
            configs.append((beta, tau))

    print(f'\n=== Sweep mode: {len(configs)} configurations ===')
    logger.info(f'Sweep mode: {len(configs)} configurations')
    for i, (beta, tau) in enumerate(configs):
        print(f'\n[Sweep {i+1}/{len(configs)}]')
        logger.info(f'Sweep {i+1}/{len(configs)}')
        run_single_configuration(args, teacher, tra_loader, val_loader, test_loader,
                                 beta, tau, logger)


def run_multiseed(args, teacher, tra_loader, val_loader, test_loader,
                  teacher_metrics_test, teacher_metrics_val, teacher_ms, logger):
    """Multi-seed run for fixed (beta, tau).

    Optionally also runs baseline (beta=0) for each seed if --multiseed_baseline.
    Writes consolidated JSON for downstream stats.
    """
    device = torch.device(args.device)

    # Truncated DDIM teacher baseline (test and val, varying NFE) -- single pass,
    # independent of seed since teacher is fixed.
    print('\n[Truncated DDIM] evaluating teacher at varying NFE')
    logger.info('[Truncated DDIM] evaluating teacher at varying NFE')
    baseline_test, baseline_val = {}, {}
    for nfe in args.nfe_grid:
        m_test = evaluate_teacher_truncated(teacher, test_loader, num_steps=nfe, device=device)
        m_val  = evaluate_teacher_truncated(teacher, val_loader,  num_steps=nfe, device=device)
        baseline_test[str(nfe)] = m_test
        baseline_val[str(nfe)]  = m_val
        print(f'  truncated NFE={nfe}: test={m_test}')

    # Configurations to run per seed
    target = (args.contrast_weight, args.contrast_temperature)
    seed_configs = [('rccd', target)]
    if args.multiseed_baseline:
        seed_configs.append(('baseline', (0.0, 0.1)))

    results = {
        'dataset': args.dataset,
        'config': {k: v for k, v in vars(args).items()
                   if isinstance(v, (int, float, str, bool, list, type(None)))},
        'teacher': {
            'full_nfe':     teacher_metrics_test,
            'full_nfe_val': teacher_metrics_val,
            'T':            args.diffusion_steps,
        },
        'baseline':     baseline_test,
        'baseline_val': baseline_val,
        'students':     {},   # {seed: {nfe_str: metrics, '_val': {...}}}  for RCCD
        'students_baseline': {},  # same shape, for the no-contrastive variant
        'latency': {},
    }

    final_student_for_latency = None

    for seed in args.seeds:
        print(f'\n========== Seed {seed} ==========')
        logger.info(f'========== Seed {seed} ==========')
        args.random_seed = seed

        for variant_name, (beta, tau) in seed_configs:
            args.contrast_weight = beta
            args.contrast_temperature = tau
            run_name = (f"seed{seed}_beta{beta}_tau{tau}" +
                        ('_baseline' if variant_name == 'baseline' else ''))

            best_student, _ = run_single_configuration(
                args, teacher, tra_loader, val_loader, test_loader,
                beta, tau, logger, run_name=run_name,
            )

            # Evaluate at all NFEs (test and val)
            seed_data = {'_run_name': run_name, '_val': {}}
            for nfe in args.nfe_grid:
                m_test = evaluate_at_nfe(best_student, test_loader, num_steps=nfe, device=device)
                m_val  = evaluate_at_nfe(best_student, val_loader,  num_steps=nfe, device=device)
                seed_data[str(nfe)]         = m_test
                seed_data['_val'][str(nfe)] = m_val
                print(f'  [{variant_name}] seed={seed} NFE={nfe} test={m_test}')

            store_key = 'students' if variant_name == 'rccd' else 'students_baseline'
            results[store_key][str(seed)] = seed_data
            final_student_for_latency = best_student

    # Latency grid: measured on one student (architecture is the same)
    if final_student_for_latency is not None:
        print('\n[Latency] measuring grid')
        sample_batch = next(iter(test_loader))
        results['latency'] = measure_latency_grid(
            teacher, final_student_for_latency, sample_batch, device, args.nfe_grid,
        )

    # Persist
    if args.out_multiseed_json is None:
        args.out_multiseed_json = os.path.join(
            'results', f'multiseed_{args.dataset}.json')
    Path(os.path.dirname(args.out_multiseed_json) or '.').mkdir(parents=True, exist_ok=True)
    with open(args.out_multiseed_json, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\n[Done] multi-seed results -> {args.out_multiseed_json}')
    logger.info(f'[Done] multi-seed results -> {args.out_multiseed_json}')


def main():
    args = parse_args()
    print(args)

    if args.sweep and args.multiseed:
        raise SystemExit('--sweep and --multiseed are mutually exclusive.')

    mode_suffix = 'sweep_' if args.sweep else ('multiseed_' if args.multiseed else '')
    logger = setup_logging(args, suffix=mode_suffix)
    logger.info(args)

    # Load data ONCE
    fix_seed(args.random_seed)
    tra_loader, val_loader, test_loader, _ = load_data(args)
    print(f'[Data] {args.dataset}: item_num={args.item_num}')
    logger.info(f'[Data] {args.dataset}: item_num={args.item_num}')

    # Teacher
    teacher, teacher_metrics_test, teacher_metrics_val, teacher_ms = \
        load_or_train_teacher(args, tra_loader, val_loader, test_loader, logger)

    # Dispatch
    if args.sweep:
        run_sweep(args, teacher, tra_loader, val_loader, test_loader, logger)
    elif args.multiseed:
        run_multiseed(args, teacher, tra_loader, val_loader, test_loader,
                      teacher_metrics_test, teacher_metrics_val, teacher_ms, logger)
    else:
        run_single_configuration(args, teacher, tra_loader, val_loader, test_loader,
                                 args.contrast_weight, args.contrast_temperature,
                                 logger)


if __name__ == '__main__':
    main()