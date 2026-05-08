"""
Multi-seed multi-variant experiment runner for the RACD pipeline.

Trains one teacher (cached on disk), then trains a *grid of student
variants* with different random seeds, evaluating each at the requested
NFE values. Also evaluates the teacher with truncated DDIM (the naive
acceleration baseline) at the same NFEs.

A "variant" is one configuration of the student loss (or other hyper-
parameters): e.g. {'vanilla_cd', 'full_racd', '+ndcg_only', '+margin_only'}.
For Block 1 ablation, one run with --variants 'vanilla_cd full_racd ndcg_only margin_only'
gives you the whole table.

Results JSON layout:
{
  'dataset':  ...,
  'config':   {...},
  'teacher':  {'full_nfe': {...metrics...}, 'T': T},
  'baseline': {nfe -> {...metrics...}},   # truncated DDIM
  'variants': {
      variant_name -> { seed -> { nfe -> {metric -> float} } }
  },
  'latency':  {
      'teacher_full': float,
      'teacher_truncated': {nfe -> float},
      'student': {variant_name -> {nfe -> float}},
  }
}

Usage:
    PYTHONPATH=../DiffuRec/src python multi_seed_runner.py \
        --dataset amazon_beauty \
        --data_root ../DiffuRec/datasets/data \
        --teacher_epochs 200 --distill_epochs 100 \
        --seeds 1997 42 2024 7 13 \
        --variants vanilla_cd full_racd \
        --nfe_grid 1 2 4 8 16 32 \
        --out_json results/beauty.json
"""
import argparse
import copy
import json
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
from distill_trainer import distill_train, evaluate_at_nfe, evaluate_teacher_full_nfe
from evaluation import evaluate_teacher_truncated, measure_latency_grid


# ---------------------------------------------------------------- #
#  Variant catalog                                                 #
#                                                                  #
#  A variant is a set of overrides applied to `args` before        #
#  building the student. Add new variants here as you design more  #
#  ablation experiments. The names below are the canonical strings #
#  used elsewhere (in JSON keys and in plotting / stats code).     #
# ---------------------------------------------------------------- #
VARIANT_CONFIGS = {
    # Block 1: RACD components
    'vanilla_cd':  dict(reward_weight=0.0, ndcg_weight=0.0,  margin_weight=0.0),
    'ndcg_only':   dict(reward_weight=1.0, ndcg_weight=0.5,  margin_weight=0.0),
    'margin_only': dict(reward_weight=1.0, ndcg_weight=0.0,  margin_weight=0.5),
    'full_racd':   dict(reward_weight=1.0, ndcg_weight=0.5,  margin_weight=0.5),

    # Block 2 placeholders — these require code-level switches in the
    # student (parametrization, solver, EMA). They map to plain kwargs
    # here; the student / trainer must read them. Until those switches
    # are wired up, requesting these variants will raise.
    'no_ema':      dict(reward_weight=1.0, ndcg_weight=0.5,  margin_weight=0.5,
                        use_ema=False),
    'with_ddim':   dict(reward_weight=1.0, ndcg_weight=0.5,  margin_weight=0.5,
                        solver='ddim'),
    'with_eps':    dict(reward_weight=1.0, ndcg_weight=0.5,  margin_weight=0.5,
                        parametrization='eps'),
}


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

    # Distillation defaults (variants override the reward fields)
    p.add_argument('--distill_lr', type=float, default=1e-3)
    p.add_argument('--distill_epochs', type=int, default=200)
    p.add_argument('--distill_eval_interval', type=int, default=5)
    p.add_argument('--distill_patience', type=int, default=10)
    p.add_argument('--cons_weight', type=float, default=1.0)
    p.add_argument('--ce_weight', type=float, default=1.0)
    p.add_argument('--ema_decay', type=float, default=0.95)
    # Default reward config — variants override these.
    p.add_argument('--reward_weight', type=float, default=0.0)
    p.add_argument('--ndcg_weight', type=float, default=0.0)
    p.add_argument('--margin_weight', type=float, default=0.0)
    p.add_argument('--ndcg_alpha', type=float, default=10.0)
    p.add_argument('--hard_neg_k', type=int, default=16)
    p.add_argument('--margin_value', type=float, default=0.0)

    # Multi-seed / multi-variant
    p.add_argument('--seeds', type=int, nargs='+', default=[1997, 42, 2024])
    p.add_argument('--teacher_seed', type=int, default=1997)
    p.add_argument('--variants', nargs='+', default=['vanilla_cd', 'full_racd'],
                   help='Variant names from VARIANT_CONFIGS to train.')
    p.add_argument('--nfe_grid', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    p.add_argument('--out_json', default='results/multiseed.json')

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


def apply_variant_overrides(base_args, variant_name):
    """Return a fresh args namespace with the variant's overrides applied."""
    if variant_name not in VARIANT_CONFIGS:
        raise KeyError(f'Unknown variant: {variant_name}. '
                       f'Known: {list(VARIANT_CONFIGS.keys())}')
    cfg = VARIANT_CONFIGS[variant_name]
    a = copy.copy(base_args)
    for k, v in cfg.items():
        setattr(a, k, v)
    return a


class _DummyLogger:
    def info(self, *a, **k): pass


def main():
    args = parse_args()
    args.save_teacher_ckpt = args.save_teacher_ckpt.format(dataset=args.dataset)
    Path(os.path.dirname(args.out_json) or '.').mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.save_teacher_ckpt) or '.').mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger = _DummyLogger()

    # ---- Load data once ----
    fix_seed(args.teacher_seed)
    tra_loader, val_loader, test_loader = load_data(args)
    print(f'[Data] {args.dataset}: item_num={args.item_num}')
    print(f'[Plan] variants={args.variants}, seeds={args.seeds}, '
          f'NFE grid={args.nfe_grid}')

    results = {
        'dataset':  args.dataset,
        'config':   {k: v for k, v in vars(args).items()
                     if isinstance(v, (int, float, str, bool, list))},
        'teacher':  {},
        'variants': {v: {} for v in args.variants},
        'baseline': {},
        'latency':  {'student': {v: {} for v in args.variants}},
    }

    # ---- Teacher (trained once, reused across variants and seeds) ----
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

    print('[Teacher] evaluating at full NFE')
    results['teacher']['full_nfe'] = evaluate_teacher_full_nfe(teacher, test_loader, device)
    results['teacher']['T'] = args.diffusion_steps

    # ---- Truncated DDIM baseline ----
    print('[Baseline] truncated DDIM at varying NFE')
    for nfe in args.nfe_grid:
        m = evaluate_teacher_truncated(teacher, test_loader, num_steps=nfe, device=device)
        results['baseline'][str(nfe)] = m
        print(f'  truncated NFE={nfe}: {m}')

    # ---- Multi-variant x multi-seed students ----
    last_student = None
    for variant_name in args.variants:
        print(f'\n=========== Variant: {variant_name} ===========')
        v_args = apply_variant_overrides(args, variant_name)
        # Surface the chosen reward weights for the log.
        print(f'  reward: weight={getattr(v_args, "reward_weight", 0)} '
              f'ndcg={getattr(v_args, "ndcg_weight", 0)} '
              f'margin={getattr(v_args, "margin_weight", 0)}')

        for seed in args.seeds:
            print(f'\n--- variant={variant_name}  seed={seed} ---')
            fix_seed(seed)

            # Reload data with this seed (reshuffles training order via DataLoader)
            tra_s, val_s, tst_s = load_data(v_args)

            student = ConsistencyStudent(teacher, v_args,
                                         ema_decay=v_args.ema_decay).to(device)
            best_student = distill_train(student, teacher, tra_s, val_s, tst_s,
                                         v_args, logger)

            # Evaluate at all NFE
            seed_results = {}
            for nfe in args.nfe_grid:
                m = evaluate_at_nfe(best_student, tst_s, num_steps=nfe, device=device)
                seed_results[str(nfe)] = m
                print(f'  variant={variant_name} seed={seed} NFE={nfe}: {m}')
            results['variants'][variant_name][str(seed)] = seed_results

            last_student = best_student  # used for latency below

        # ---- Latency for this variant (one batch is enough) ----
        sample_batch = next(iter(test_loader))
        lat = measure_latency_grid(teacher, last_student, sample_batch, device,
                                   args.nfe_grid)
        results['latency']['student'][variant_name] = lat['student']
        # teacher_full / teacher_truncated are variant-independent
        results['latency'].setdefault('teacher_full', lat['teacher_full'])
        results['latency'].setdefault('teacher_truncated', lat['teacher_truncated'])

    # ---- Persist ----
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\n[Done] results -> {args.out_json}')


if __name__ == '__main__':
    main()