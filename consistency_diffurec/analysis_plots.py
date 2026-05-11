"""
Method-analysis figures.

Three figures:
  1. t-SNE of item embeddings (teacher vs student)
  2. Denoising trajectory — fixed sample index for reproducibility
  3. SVD spectrum + subspace angles between top-K singular directions

Run from `consistency_diffurec/`:
    python analysis_plots.py \
        --dataset toys \
        --data_root /content/diffurec-distillation/datasets/data \
        --teacher_ckpt checkpoints/teacher_toys.pt \
        --student_ckpt checkpoints/student_rccd_toys.pt \
        --trajectory_sample_idx 0 \
        --out_dir figures/
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from utils import Data_Test
from model import create_model_diffu, Att_Diffuse_model
from consistency_diffurec import ConsistencyStudent
from plots import COLORS, MARKERS, apply_style, _save


def _build_args_for_loading(dataset, data_root, max_len=50):
    class _A: pass
    a = _A()
    a.dataset = dataset
    a.data_root = data_root
    a.max_len = max_len
    a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a.batch_size = 256
    a.hidden_size = 128
    a.dropout = 0.1
    a.emb_dropout = 0.3
    a.hidden_act = 'gelu'
    a.num_blocks = 4
    a.diffusion_steps = 32
    a.lambda_uncertainty = 0.001
    a.noise_schedule = 'trunc_lin'
    a.rescale_timesteps = True
    a.schedule_sampler_name = 'lossaware'
    return a


def _load_models_and_data(args_ns):
    path = os.path.join(args_ns.data_root, args_ns.dataset, 'dataset.pkl')
    with open(path, 'rb') as f:
        data_raw = pickle.load(f)
    args_ns.item_num = len(data_raw['smap'])

    device = torch.device(args_ns.device)
    teacher = Att_Diffuse_model(create_model_diffu(args_ns), args_ns).to(device)
    teacher.load_state_dict(torch.load(args_ns.teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = ConsistencyStudent(teacher, args_ns).to(device)
    student.load_state_dict(torch.load(args_ns.student_ckpt, map_location=device))
    student.eval()

    test_loader = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'],
                            args_ns).get_pytorch_dataloaders()
    return teacher, student, test_loader, data_raw


def plot_tsne_embeddings(teacher, student, n_items=2000, n_clusters=8,
                         out_path='figures/tsne_embeddings.pdf', seed=0):
    apply_style()
    # Full seeding for reproducibility
    np.random.seed(seed)
    with torch.no_grad():
        e_teacher = teacher.item_embeddings.weight.detach().cpu().numpy()
        e_student = student.item_embeddings.weight.detach().cpu().numpy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(min(e_teacher.shape[0], e_student.shape[0]),
                     size=min(n_items, e_teacher.shape[0]), replace=False)
    et = e_teacher[idx]
    es = e_student[idx]

    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(et)

    print('[t-SNE] computing teacher projection...')
    proj_t = TSNE(n_components=2, random_state=seed, init='pca',
                  perplexity=30, learning_rate='auto').fit_transform(et)
    print('[t-SNE] computing student projection...')
    proj_s = TSNE(n_components=2, random_state=seed, init='pca',
                  perplexity=30, learning_rate='auto').fit_transform(es)

    cmap = plt.get_cmap('cividis')(np.linspace(0.05, 0.95, n_clusters))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, proj, title in [(axes[0], proj_t, 'Teacher item embeddings'),
                            (axes[1], proj_s, 'Student item embeddings')]:
        for c in range(n_clusters):
            m = labels == c
            ax.scatter(proj[m, 0], proj[m, 1], s=8, alpha=0.7,
                       color=cmap[c], edgecolor='none')
        ax.set_title(title); ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('Embedding structure preservation under consistency distillation', y=1.02)
    _save(fig, out_path)


@torch.no_grad()
def _teacher_full_trajectory(teacher, sequence, device):
    diffu = teacher.diffu
    item_emb = teacher.item_embeddings(sequence)
    item_emb = teacher.embed_dropout(item_emb)
    item_emb = teacher.LayerNorm(item_emb)
    mask_seq = (sequence > 0).float()

    bs = sequence.size(0)
    H = item_emb.size(-1)
    T = diffu.num_timesteps

    x_t = torch.randn(bs, H, device=device)
    trajectory = []
    indices = list(range(T))[::-1]
    for i in indices:
        t = torch.tensor([i] * bs, device=device, dtype=torch.long)
        x_0_pred, _ = diffu.xstart_model(item_emb, x_t, diffu._scale_timesteps(t), mask_seq)
        trajectory.append(x_0_pred[0].cpu().numpy())
        x_t = diffu.p_sample(item_emb, x_t, t, mask_seq)
    return np.stack(trajectory)


@torch.no_grad()
def _student_single_step(student, sequence, device):
    item_rep, mask_seq = student.encode(sequence)
    bs = sequence.size(0)
    H = item_rep.size(-1)
    T = student.diffu_student.num_timesteps
    x_t = torch.randn(bs, H, device=device)
    t = torch.full((bs,), T - 1, device=device, dtype=torch.long)
    x_0 = student.diffu_student.predict_x0(item_rep, x_t, t, mask_seq)
    return x_0[0].cpu().numpy()


def plot_denoising_trajectory(teacher, student, test_loader, device,
                              sample_idx=0,
                              out_path='figures/denoising_trajectory.pdf',
                              seed=0):
    """
    Use FIXED sample_idx for reproducibility (sample from first batch).
    """
    apply_style()
    # Seed RNG so trajectory noise is reproducible
    torch.manual_seed(seed)
    np.random.seed(seed)

    sample_seq, _ = next(iter(test_loader))
    if sample_idx >= sample_seq.size(0):
        print(f'[trajectory] sample_idx={sample_idx} out of range, using 0')
        sample_idx = 0
    sample_seq = sample_seq[sample_idx:sample_idx + 1].to(device)
    print(f'[trajectory] using fixed sample_idx={sample_idx}')

    target_idx = sample_seq[0, -1].item()
    if target_idx == 0:
        nz = sample_seq[0].nonzero().flatten()
        target_idx = sample_seq[0, nz[-1]].item()
    e_target = teacher.item_embeddings.weight[target_idx].detach().cpu().numpy()

    print('[trajectory] running teacher full reverse loop...')
    traj_teacher = _teacher_full_trajectory(teacher, sample_seq, device)
    print('[trajectory] running student single step...')
    pred_student = _student_single_step(student, sample_seq, device)

    from sklearn.decomposition import PCA
    all_pts = np.vstack([traj_teacher, pred_student[None, :], e_target[None, :]])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_pts)
    proj_teacher = proj[:len(traj_teacher)]
    proj_student = proj[len(traj_teacher)]
    proj_target  = proj[-1]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(proj_teacher[:, 0], proj_teacher[:, 1],
            color=COLORS['teacher'], linewidth=1.5, alpha=0.7,
            label='Teacher trajectory (32 steps)')
    n_pts = len(proj_teacher)
    sc = ax.scatter(proj_teacher[:, 0], proj_teacher[:, 1],
                    c=np.arange(n_pts), cmap='cividis', s=30,
                    edgecolor='black', linewidth=0.4, zorder=3)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label('Reverse step (T-1 → 0)')

    ax.scatter(proj_teacher[0, 0], proj_teacher[0, 1],
               marker='X', s=180, color=COLORS['baseline'],
               edgecolor='black', linewidth=0.8, zorder=4,
               label='Teacher start (NFE=T)')
    ax.scatter(proj_teacher[-1, 0], proj_teacher[-1, 1],
               marker='*', s=240, color=COLORS['teacher'],
               edgecolor='black', linewidth=0.8, zorder=4,
               label='Teacher end (predicted x₀)')
    ax.scatter(proj_student[0], proj_student[1],
               marker='o', s=200, color=COLORS['student'],
               edgecolor='black', linewidth=0.8, zorder=5,
               label='Student (NFE=1)')
    ax.scatter(proj_target[0], proj_target[1],
               marker='D', s=140, color=COLORS['tertiary'],
               edgecolor='black', linewidth=0.8, zorder=4,
               label='True target embedding')
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
    ax.set_title(f'Denoising trajectory (fixed sample idx={sample_idx})')
    ax.legend(loc='best', fontsize=9)
    _save(fig, out_path)


def plot_svd_spectrum(teacher, student, out_path='figures/svd_spectrum.pdf', top_k=10):
    """
    SVD spectrum + subspace angles between top-K singular directions.
    Subspace angles give the geometric difference between teacher's and
    student's principal embedding directions — addresses the criticism
    that singular values alone could miss a rotation.
    """
    apply_style()
    from scipy.linalg import subspace_angles

    with torch.no_grad():
        et = teacher.item_embeddings.weight.detach().cpu().numpy()
        es = student.item_embeddings.weight.detach().cpu().numpy()

    Ut, s_t, _  = np.linalg.svd(et, full_matrices=False)
    Us, s_s, _  = np.linalg.svd(es, full_matrices=False)

    # Subspace angles between the top-K right-singular subspaces.
    # (Use left singular vectors of the M x D embedding matrix.)
    k = min(top_k, Ut.shape[1], Us.shape[1])
    angles_rad = subspace_angles(Ut[:, :k], Us[:, :k])
    angles_deg = np.rad2deg(angles_rad)
    max_angle = float(angles_deg.max())
    mean_angle = float(angles_deg.mean())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(np.arange(len(s_t)), s_t, color=COLORS['teacher'],
            linewidth=2.0, label='Teacher')
    ax.plot(np.arange(len(s_s)), s_s, color=COLORS['student'],
            linewidth=2.0, linestyle='--', label='Student')
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Singular value (linear)')
    ax.set_title('Spectrum (linear)')
    ax.legend()

    ax = axes[1]
    ax.semilogy(np.arange(len(s_t)), s_t, color=COLORS['teacher'],
                linewidth=2.0, label='Teacher')
    ax.semilogy(np.arange(len(s_s)), s_s, color=COLORS['student'],
                linewidth=2.0, linestyle='--', label='Student')
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Singular value (log scale)')
    ax.set_title('Spectrum (log)')
    ax.legend()

    ax = axes[2]
    ax.bar(np.arange(k), angles_deg, color=COLORS['student'],
           edgecolor='black', linewidth=0.5)
    ax.axhline(y=90, color='red', linestyle=':', alpha=0.5, label='90° (orthogonal)')
    ax.set_xlabel('Principal direction index')
    ax.set_ylabel('Subspace angle (degrees)')
    ax.set_title(f'Top-{k} subspace angles\n(0° = aligned, 90° = orthogonal)')
    ax.set_ylim(0, 95)
    ax.legend(fontsize=8)

    cos_sim_spectrum = np.dot(s_t, s_s) / (np.linalg.norm(s_t) * np.linalg.norm(s_s))
    rel_diff_spectrum = np.linalg.norm(s_t - s_s) / np.linalg.norm(s_t)
    fig.suptitle(
        f'Embedding spectrum: cos similarity = {cos_sim_spectrum:.4f}, '
        f'rel L2 diff = {rel_diff_spectrum:.4f}. '
        f'Top-{k} subspace angles: max = {max_angle:.1f}°, mean = {mean_angle:.1f}°',
        y=1.02, fontsize=10,
    )
    _save(fig, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--data_root', default='../data/datasets')

    # Two ways to specify checkpoints:
    # (1) explicit paths
    p.add_argument('--teacher_ckpt', default=None)
    p.add_argument('--student_ckpt', default=None)
    # (2) run_name — auto-resolves to artifacts/<dataset>/<run_name>/student_final.pt
    #     and artifacts/<dataset>/teacher/teacher.pt
    p.add_argument('--run_name', default=None,
                   help='If set, auto-resolves checkpoint paths from artifacts/.')

    p.add_argument('--max_len', type=int, default=50)
    p.add_argument('--out_dir', default='figures')
    p.add_argument('--n_items_tsne', type=int, default=2000)
    p.add_argument('--n_clusters_tsne', type=int, default=8)
    p.add_argument('--trajectory_sample_idx', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--svd_top_k', type=int, default=10)
    args_cli = p.parse_args()

    # Auto-resolve checkpoints if --run_name was given
    if args_cli.run_name is not None:
        if args_cli.teacher_ckpt is None:
            args_cli.teacher_ckpt = os.path.join('artifacts', args_cli.dataset,
                                                 'teacher', 'teacher.pt')
        if args_cli.student_ckpt is None:
            args_cli.student_ckpt = os.path.join('artifacts', args_cli.dataset,
                                                 args_cli.run_name, 'student_final.pt')

    if args_cli.teacher_ckpt is None or args_cli.student_ckpt is None:
        raise SystemExit('Must provide either --run_name or both '
                         '--teacher_ckpt and --student_ckpt.')

    args_ns = _build_args_for_loading(args_cli.dataset, args_cli.data_root,
                                      max_len=args_cli.max_len)
    args_ns.teacher_ckpt = args_cli.teacher_ckpt
    args_ns.student_ckpt = args_cli.student_ckpt

    Path(args_cli.out_dir).mkdir(parents=True, exist_ok=True)
    teacher, student, test_loader, _ = _load_models_and_data(args_ns)

    slug = args_cli.dataset.replace('/', '_')
    suffix = f'_{args_cli.run_name}' if args_cli.run_name else ''

    plot_tsne_embeddings(
        teacher, student,
        n_items=args_cli.n_items_tsne, n_clusters=args_cli.n_clusters_tsne,
        out_path=os.path.join(args_cli.out_dir, f'tsne_embeddings_{slug}{suffix}.pdf'),
        seed=args_cli.seed,
    )
    plot_denoising_trajectory(
        teacher, student, test_loader, torch.device(args_ns.device),
        sample_idx=args_cli.trajectory_sample_idx,
        out_path=os.path.join(args_cli.out_dir, f'denoising_trajectory_{slug}{suffix}.pdf'),
        seed=args_cli.seed,
    )
    plot_svd_spectrum(
        teacher, student,
        out_path=os.path.join(args_cli.out_dir, f'svd_spectrum_{slug}{suffix}.pdf'),
        top_k=args_cli.svd_top_k,
    )

    print(f'\nAll three analysis figures written to {args_cli.out_dir}/')


if __name__ == '__main__':
    main()