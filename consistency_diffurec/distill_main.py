"""
Entry point for consistency distillation of DiffuRec.

Step 1: train (or load) the DiffuRec teacher.
Step 2: initialize a student from the teacher.
Step 3: train the student via consistency distillation.
Step 4: evaluate the student at NFE = 1, 2, 4, 8 and compare to the
        teacher (NFE = T = diffusion_steps).

Run from the `src/` directory.
"""
import argparse
import logging
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import Data_Train, Data_Val, Data_Test
from model import create_model_diffu, Att_Diffuse_model
from trainer import model_train

from consistency_diffurec import ConsistencyStudent
from distill_trainer import (
    distill_train,
    evaluate_teacher_full_nfe,
    measure_inference_latency,
)


def parse_args():
    p = argparse.ArgumentParser()
    # ----- DiffuRec args (mirror main.py) -----
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
    p.add_argument('--description', type=str, default='Distill')
    p.add_argument('--long_head', default=False)
    p.add_argument('--diversity_measure', default=False)
    p.add_argument('--epoch_time_avg', default=False)

    # ----- Distillation-specific -----
    p.add_argument('--teacher_ckpt', type=str, default=None,
                   help='Path to a saved teacher state_dict. If absent, the '
                        'teacher is trained from scratch.')
    p.add_argument('--save_teacher_ckpt', type=str, default='checkpoints/teacher.pt')
    p.add_argument('--save_student_ckpt', type=str, default='checkpoints/student.pt')
    p.add_argument('--distill_lr', type=float, default=1e-3)
    p.add_argument('--distill_epochs', type=int, default=200)
    p.add_argument('--distill_eval_interval', type=int, default=5)
    p.add_argument('--distill_patience', type=int, default=10)
    p.add_argument('--cons_weight', type=float, default=1.0)
    p.add_argument('--ce_weight', type=float, default=1.0)
    p.add_argument('--ema_decay', type=float, default=0.95)
    p.add_argument('--data_root', type=str, default='../datasets/data',
                   help='Override the dataset root directory if your layout differs.')

    return p.parse_args()


def fix_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    cudnn.deterministic = True
    cudnn.benchmark = False


def setup_logging(args):
    log_dir = os.path.join(args.log_file, args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    fname = os.path.join(log_dir, 'distill_' + time.strftime('%Y-%m-%d_%H-%M-%S') + '.log')
    logging.basicConfig(level=logging.INFO, filename=fname,
                        format='%(asctime)s - %(message)s', filemode='w')
    return logging.getLogger(__name__)


def load_data(args):
    path = os.path.join(args.data_root, args.dataset, 'dataset.pkl')
    with open(path, 'rb') as f:
        data_raw = pickle.load(f)
    args.item_num = len(data_raw['smap'])
    tra = Data_Train(data_raw['train'], args).get_pytorch_dataloaders()
    val = Data_Val(data_raw['train'], data_raw['val'], args).get_pytorch_dataloaders()
    tst = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args).get_pytorch_dataloaders()
    return tra, val, tst


def main():
    args = parse_args()
    print(args)
    fix_seed(args.random_seed)
    logger = setup_logging(args)
    logger.info(args)

    tra_loader, val_loader, test_loader = load_data(args)

    # === Step 1: teacher ===
    teacher = Att_Diffuse_model(create_model_diffu(args), args).to(args.device)

    if args.teacher_ckpt and os.path.exists(args.teacher_ckpt):
        print(f'Loading teacher from {args.teacher_ckpt}')
        logger.info(f'Loading teacher from {args.teacher_ckpt}')
        teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=args.device))
    else:
        print('Training teacher from scratch...')
        logger.info('Training teacher from scratch...')
        teacher, _ = model_train(tra_loader, val_loader, test_loader, teacher, args, logger)
        os.makedirs(os.path.dirname(args.save_teacher_ckpt) or '.', exist_ok=True)
        torch.save(teacher.state_dict(), args.save_teacher_ckpt)
        print(f'Teacher saved to {args.save_teacher_ckpt}')
        logger.info(f'Teacher saved to {args.save_teacher_ckpt}')

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Reference: teacher at full NFE.
    print('\n=== Teacher (NFE=T) Test Performance ===')
    logger.info('=== Teacher (NFE=T) Test Performance ===')
    teacher_metrics = evaluate_teacher_full_nfe(teacher, test_loader, torch.device(args.device))
    print(teacher_metrics)
    logger.info(teacher_metrics)
    sample_batch = next(iter(test_loader))
    teacher_ms = measure_inference_latency(
        teacher, sample_batch, torch.device(args.device), mode='teacher'
    )
    print(f'Teacher latency (NFE=T): {teacher_ms:.4f} ms/sample')
    logger.info(f'Teacher latency (NFE=T): {teacher_ms:.4f} ms/sample')

    # === Step 2 & 3: student via consistency distillation ===
    print('\n=== Initializing student ===')
    logger.info('=== Initializing student ===')
    student = ConsistencyStudent(teacher, args, ema_decay=args.ema_decay)

    print('\n=== Running consistency distillation ===')
    logger.info('=== Running consistency distillation ===')
    best_student = distill_train(
        student, teacher.diffu, tra_loader, val_loader, test_loader, args, logger
    )

    os.makedirs(os.path.dirname(args.save_student_ckpt) or '.', exist_ok=True)
    torch.save(best_student.state_dict(), args.save_student_ckpt)
    print(f'Student saved to {args.save_student_ckpt}')
    logger.info(f'Student saved to {args.save_student_ckpt}')


if __name__ == '__main__':
    main()