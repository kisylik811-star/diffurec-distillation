"""Training loop and evaluation utilities for consistency distillation (RACD)."""
import copy
import time
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


def distill_train(student, teacher_model, train_loader, val_loader, test_loader,
                  args, logger):
    """
    Train the student via (Reward-Aware) Consistency Distillation against a
    frozen teacher.

    `teacher_model` is the full Att_Diffuse_model (not just teacher_diffu),
    because reward mining (margin term) needs teacher.item_embeddings as well
    as teacher.diffu.
    """
    device = torch.device(args.device)
    student = student.to(device)

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_diffu = teacher_model.diffu

    optimizer = optim.Adam(student.parameters(), lr=args.distill_lr)

    best_score = -1.0
    best_student = None
    bad_count = 0

    cons_w   = float(getattr(args, 'cons_weight', 1.0))
    ce_w     = float(getattr(args, 'ce_weight', 1.0))
    reward_w = float(getattr(args, 'reward_weight', 1.0))

    for epoch in range(args.distill_epochs):
        student.train()
        running_cons, running_ce, running_rwd, n_b = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            seq, target = [x.to(device) for x in batch]
            optimizer.zero_grad()
            cons_loss, ce_loss, reward_loss = student.consistency_loss(
                seq, target, teacher_diffu, teacher_model=teacher_model
            )
            loss = cons_w * cons_loss + ce_w * ce_loss + reward_w * reward_loss
            loss.backward()
            optimizer.step()
            student.update_ema()
            running_cons += float(cons_loss.detach())
            running_ce   += float(ce_loss.detach())
            running_rwd  += float(reward_loss.detach())
            n_b += 1

        msg = (f'[Distill][Epoch {epoch}] '
               f'cons={running_cons / max(n_b, 1):.4f} '
               f'ce={running_ce / max(n_b, 1):.4f} '
               f'reward={running_rwd / max(n_b, 1):.4f}')
        print(msg)
        logger.info(msg)

        if epoch % args.distill_eval_interval == 0:
            val_metrics = evaluate_at_nfe(student, val_loader, num_steps=1, device=device)
            msg = f'[Val NFE=1] {val_metrics}'
            print(msg)
            logger.info(msg)

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