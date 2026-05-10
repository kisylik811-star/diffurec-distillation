"""Training loop and evaluation utilities for consistency distillation.

Updated for RCCD: tracks three loss components (cons, ce, contrast) and
writes them to the per-epoch CSV log.
"""
import copy
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from trainer import hrs_and_ndcgs_k


@torch.no_grad()
def evaluate_at_nfe(student, loader, num_steps, device, ks=(5, 10, 20)):
    """Evaluate the student at a given number of forward passes (NFE)."""
    student.eval()
    acc = {f'HR@{k}': [] for k in ks}
    acc.update({f'NDCG@{k}': [] for k in ks})
    for batch in loader:
        seq, target = [x.to(device) for x in batch]
        scores = student.predict_scores(seq, num_steps=num_steps)
        m = hrs_and_ndcgs_k(scores, target, list(ks))
        for k, v in m.items():
            acc[k].append(v)
    return {k: round(float(np.mean(v)) * 100, 4) for k, v in acc.items()}


@torch.no_grad()
def evaluate_teacher_full_nfe(teacher_model, loader, device, ks=(5, 10, 20)):
    """Evaluate the teacher with its native iterative reverse loop (NFE = T)."""
    teacher_model.eval()
    acc = {f'HR@{k}': [] for k in ks}
    acc.update({f'NDCG@{k}': [] for k in ks})
    for batch in loader:
        seq, target = [x.to(device) for x in batch]
        _, rep_diffu, _, _, _, _ = teacher_model(seq, target, train_flag=False)
        scores = teacher_model.diffu_rep_pre(rep_diffu)
        m = hrs_and_ndcgs_k(scores, target, list(ks))
        for k, v in m.items():
            acc[k].append(v)
    return {k: round(float(np.mean(v)) * 100, 4) for k, v in acc.items()}


@torch.no_grad()
def measure_inference_latency(model, sample_batch, device, num_steps=None,
                              n_warmup=10, n_runs=50, mode='student'):
    """Measure mean inference time per sample (in milliseconds)."""
    seq, target = [x.to(device) for x in sample_batch]
    model.eval()

    def _run_once():
        if mode == 'student':
            return model.predict_scores(seq, num_steps=num_steps)
        else:
            _, rep_diffu, _, _, _, _ = model(seq, target, train_flag=False)
            return model.diffu_rep_pre(rep_diffu)

    for _ in range(n_warmup):
        _ = _run_once()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        _ = _run_once()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - t0) * 1000.0 / n_runs
    return elapsed_ms / seq.size(0)


def _open_csv(path, header):
    """Create a CSV file at `path`, write the header. Returns (handle, writer)."""
    Path(os.path.dirname(path) or '.').mkdir(parents=True, exist_ok=True)
    f = open(path, 'w', newline='')
    w = csv.writer(f)
    w.writerow(header)
    return f, w


def distill_train(student, teacher_diffu, train_loader, val_loader, test_loader,
                  args, logger, log_csv_path=None):
    """Train the student via RCCD against a frozen teacher.

    Two CSV files are produced when `log_csv_path` is set:
      - `<log_csv_path>`         -> per-epoch cons/ce/contrast/total loss
      - `<log_csv_path>.val.csv` -> per-eval val HR/NDCG at NFE=1
    """
    device = torch.device(args.device)
    student = student.to(device)

    teacher_diffu.eval()
    for p in teacher_diffu.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(student.parameters(), lr=args.distill_lr)

    # Pull weights with safe defaults so older configs still work.
    contrast_weight = getattr(args, 'contrast_weight', 0.5)
    contrast_temperature = getattr(args, 'contrast_temperature', 0.1)

    loss_writer = val_writer = None
    loss_f = val_f = None
    if log_csv_path is not None:
        loss_f, loss_writer = _open_csv(
            log_csv_path,
            ['epoch', 'cons_loss', 'ce_loss', 'contrast_loss', 'total_loss'],
        )
        val_path = log_csv_path + '.val.csv'
        val_f, val_writer = _open_csv(
            val_path,
            ['epoch', 'HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20'],
        )

    best_score = -1.0
    best_student = None
    bad_count = 0

    try:
        for epoch in range(args.distill_epochs):
            student.train()
            running_cons, running_ce, running_contrast, n_b = 0.0, 0.0, 0.0, 0
            for batch in train_loader:
                seq, target = [x.to(device) for x in batch]
                optimizer.zero_grad()

                cons_loss, ce_loss, contrast_loss = student.consistency_loss(
                    seq, target, teacher_diffu,
                    cons_weight=args.cons_weight,
                    ce_weight=args.ce_weight,
                    contrast_weight=contrast_weight,
                    contrast_temperature=contrast_temperature,
                )
                loss = (args.cons_weight * cons_loss
                        + args.ce_weight * ce_loss
                        + contrast_weight * contrast_loss)

                loss.backward()
                optimizer.step()
                student.update_ema()

                running_cons += cons_loss.item()
                running_ce += ce_loss.item()
                running_contrast += contrast_loss.item()
                n_b += 1

            avg_cons = running_cons / max(n_b, 1)
            avg_ce = running_ce / max(n_b, 1)
            avg_contrast = running_contrast / max(n_b, 1)
            avg_total = (args.cons_weight * avg_cons
                         + args.ce_weight * avg_ce
                         + contrast_weight * avg_contrast)

            msg = (f'[Distill][Epoch {epoch}] cons={avg_cons:.4f} '
                   f'ce={avg_ce:.4f} contrast={avg_contrast:.4f}')
            print(msg)
            logger.info(msg)

            if loss_writer is not None:
                loss_writer.writerow([epoch, avg_cons, avg_ce, avg_contrast, avg_total])
                loss_f.flush()

            if epoch % args.distill_eval_interval == 0:
                val_metrics = evaluate_at_nfe(student, val_loader, num_steps=1, device=device)
                msg = f'[Val NFE=1] {val_metrics}'
                print(msg)
                logger.info(msg)

                if val_writer is not None:
                    val_writer.writerow([
                        epoch,
                        val_metrics['HR@5'], val_metrics['HR@10'], val_metrics['HR@20'],
                        val_metrics['NDCG@5'], val_metrics['NDCG@10'], val_metrics['NDCG@20'],
                    ])
                    val_f.flush()

                if val_metrics['HR@10'] > best_score:
                    best_score = val_metrics['HR@10']
                    best_student = copy.deepcopy(student)
                    bad_count = 0
                else:
                    bad_count += 1
                    if bad_count >= args.distill_patience:
                        print('Early stop')
                        logger.info('Early stop')
                        break
    finally:
        if loss_f is not None:
            loss_f.close()
        if val_f is not None:
            val_f.close()

    if best_student is None:
        best_student = student

    print('\n=== Final Test (Student, varying NFE) ===')
    logger.info('=== Final Test (Student, varying NFE) ===')
    for nfe in [1, 2, 4, 8]:
        m = evaluate_at_nfe(best_student, test_loader, num_steps=nfe, device=device)
        line = f'  NFE={nfe} {m}'
        print(line)
        logger.info(line)

    print('\n=== Inference Latency (ms/sample) ===')
    logger.info('=== Inference Latency (ms/sample) ===')
    sample_batch = next(iter(test_loader))
    for nfe in [1, 2, 4, 8]:
        ms = measure_inference_latency(best_student, sample_batch, device,
                                       num_steps=nfe, mode='student')
        line = f'  Student NFE={nfe}: {ms:.4f}'
        print(line)
        logger.info(line)

    return best_student