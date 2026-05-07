"""
Method-analysis figures.

These plots require loaded teacher and student model checkpoints (not just
metrics JSONs), so they live in a separate script from `plots.py`.

Three figures:
  1. t-SNE of item embeddings: teacher vs student, coloured by class.
     Shows that distillation preserved the embedding-space structure.

  2. Denoising-trajectory visualization: for one test sequence, show how the
     teacher's predicted x_0 evolves across its 32 reverse steps, and where
     the student's single-shot prediction lands. Visually demonstrates that
     the student takes one jump to the teacher's endpoint.

  3. Singular-value spectrum of the item embedding matrix: teacher vs student.
     Quantitative analog of "spectral analysis" — shows that distillation
     preserved the principal directions of the embedding space.

Run from `consistency_diffurec/`:

    PYTHONPATH=../original_diffurec python analysis_plots.py \
        --dataset amazon_beauty \
        --data_root ../data/datasets \
        --teacher_ckpt checkpoints/teacher_amazon_beauty.pt \
        --student_ckpt checkpoints/student.pt \
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


# --------------------------------------------------------------------- #
#  Loading helpers                                                      #
# --------------------------------------------------------------------- #
def _build_args_for_loading(dataset, data_root, max_len=50):
    """Reconstruct minimal args object for model construction."""
    class _A:
        pass
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


# --------------------------------------------------------------------- #
#  Figure 1: t-SNE of item embeddings                                   #
# --------------------------------------------------------------------- #
def plot_tsne_embeddings(teacher, student, n_items=2000, n_clusters=8,
                         out_path='figures/tsne_embeddings.pdf', seed=0):
    """
    Project teacher and student item embeddings into 2D with t-SNE.
    Cluster the teacher's embedding (k-means) for colouring and reuse the
    same labels on the student plot to make structural similarity visible.
    """
    apply_style()
    with torch.no_grad():
        e_teacher = teacher.item_embeddings.weight.detach().cpu().numpy()
        e_student = student.item_embeddings.weight.detach().cpu().numpy()

    # Subsample a fixed set of items so both panels show the same items
    rng = np.random.default_rng(seed)
    idx = rng.choice(min(e_teacher.shape[0], e_student.shape[0]),
                     size=min(n_items, e_teacher.shape[0]), replace=False)
    et = e_teacher[idx]
    es = e_student[idx]

    # Cluster teacher embeddings; use those labels as a visual anchor for both
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(et)

    print('[t-SNE] computing teacher projection...')
    proj_t = TSNE(n_components=2, random_state=seed, init='pca',
                  perplexity=30, learning_rate='auto').fit_transform(et)
    print('[t-SNE] computing student projection...')
    proj_s = TSNE(n_components=2, random_state=seed, init='pca',
                  perplexity=30, learning_rate='auto').fit_transform(es)

    # Build a black-yellow-brown discrete palette for clusters
    cmap = plt.get_cmap('cividis')(np.linspace(0.05, 0.95, n_clusters))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, proj, title in [
        (axes[0], proj_t, 'Teacher item embeddings'),
        (axes[1], proj_s, 'Student item embeddings'),
    ]:
        for c in range(n_clusters):
            m = labels == c
            ax.scatter(proj[m, 0], proj[m, 1], s=8, alpha=0.7,
                       color=cmap[c], edgecolor='none')
        ax.set_title(title)
        ax.set_xlabel('t-SNE dim 1')
        ax.set_ylabel('t-SNE dim 2')
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('Embedding structure preservation under consistency distillation',
                 y=1.02)
    _save(fig, out_path)


# --------------------------------------------------------------------- #
#  Figure 2: Denoising trajectory                                       #
# --------------------------------------------------------------------- #
@torch.no_grad()
def _teacher_full_trajectory(teacher, sequence, device):
    """Run the teacher's full reverse loop and record predicted x_0 at each step."""
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
        trajectory.append(x_0_pred[0].cpu().numpy())  # one example
        x_t = diffu.p_sample(item_emb, x_t, t, mask_seq)
    return np.stack(trajectory)  # (T, H)


@torch.no_grad()
def _student_single_step(student, sequence, device):
    item_rep, mask_seq = student.encode(sequence)
    bs = sequence.size(0)
    H = item_rep.size(-1)
    T = student.diffu_student.num_timesteps
    x_t = torch.randn(bs, H, device=device)
    t = torch.full((bs,), T - 1, device=device, dtype=torch.long)
    x_0 = student.diffu_student.predict_x0(item_rep, x_t, t, mask_seq)
    return x_0[0].cpu().numpy()  # (H,)


def plot_denoising_trajectory(teacher, student, test_loader, device,
                              out_path='figures/denoising_trajectory.pdf'):
    """
    For one test sample: project teacher's 32-step trajectory of predicted
    x_0 and student's single-shot prediction into 2D (PCA on stacked points)
    and draw the trajectory. Goal: visually show that the student lands near
    the teacher's converged endpoint in one step.
    """
    apply_style()
    sample_seq, _ = next(iter(test_loader))
    sample_seq = sample_seq[:1].to(device)  # one sample

    # Get true target embedding for reference
    target_idx = sample_seq[0, -1].item()
    if target_idx == 0:
        # last position is padding, take last non-zero
        nz = sample_seq[0].nonzero().flatten()
        target_idx = sample_seq[0, nz[-1]].item()
    e_target = teacher.item_embeddings.weight[target_idx].detach().cpu().numpy()

    print('[trajectory] running teacher full reverse loop...')
    traj_teacher = _teacher_full_trajectory(teacher, sample_seq, device)
    print('[trajectory] running student single step...')
    pred_student = _student_single_step(student, sample_seq, device)

    # PCA on all points stacked together, so projections are comparable
    from sklearn.decomposition import PCA
    all_pts = np.vstack([traj_teacher, pred_student[None, :], e_target[None, :]])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_pts)
    proj_teacher  = proj[:len(traj_teacher)]
    proj_student  = proj[len(traj_teacher)]
    proj_target   = proj[-1]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    # Teacher trajectory
    ax.plot(proj_teacher[:, 0], proj_teacher[:, 1],
            color=COLORS['teacher'], linewidth=1.5, alpha=0.7,
            label='Teacher trajectory (32 steps)')
    # Mark each step
    n_pts = len(proj_teacher)
    sc = ax.scatter(proj_teacher[:, 0], proj_teacher[:, 1],
                    c=np.arange(n_pts), cmap='cividis', s=30,
                    edgecolor='black', linewidth=0.4, zorder=3)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label('Reverse step (T-1 → 0)')

    # Endpoints
    ax.scatter(proj_teacher[0, 0], proj_teacher[0, 1],
               marker='X', s=180, color=COLORS['baseline'],
               edgecolor='black', linewidth=0.8, zorder=4,
               label='Teacher start (NFE=T)')
    ax.scatter(proj_teacher[-1, 0], proj_teacher[-1, 1],
               marker='*', s=240, color=COLORS['teacher'],
               edgecolor='black', linewidth=0.8, zorder=4,
               label='Teacher end (predicted x₀)')

    # Student single shot
    ax.scatter(proj_student[0], proj_student[1],
               marker='o', s=200, color=COLORS['student'],
               edgecolor='black', linewidth=0.8, zorder=5,
               label='Student (NFE=1)')

    # Ground truth target embedding
    ax.scatter(proj_target[0], proj_target[1],
               marker='D', s=140, color=COLORS['tertiary'],
               edgecolor='black', linewidth=0.8, zorder=4,
               label='True target embedding')

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('Denoising trajectory: teacher (32 steps) vs student (1 step)')
    ax.legend(loc='best', fontsize=9)
    _save(fig, out_path)


# --------------------------------------------------------------------- #
#  Figure 3: SVD spectrum of embedding matrices                         #
# --------------------------------------------------------------------- #
def plot_svd_spectrum(teacher, student, out_path='figures/svd_spectrum.pdf'):
    """
    Singular-value spectrum of the item embedding matrices.
    Closer spectra = the student preserved the principal directions of the
    teacher's embedding space.
    """
    apply_style()
    with torch.no_grad():
        et = teacher.item_embeddings.weight.detach().cpu().numpy()
        es = student.item_embeddings.weight.detach().cpu().numpy()

    s_t = np.linalg.svd(et, compute_uv=False)
    s_s = np.linalg.svd(es, compute_uv=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Linear scale
    ax = axes[0]
    ax.plot(np.arange(len(s_t)), s_t, color=COLORS['teacher'],
            linewidth=2.0, label='Teacher')
    ax.plot(np.arange(len(s_s)), s_s, color=COLORS['student'],
            linewidth=2.0, linestyle='--', label='Student')
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Singular value')
    ax.set_title('Spectrum (linear scale)')
    ax.legend()

    # Log scale — exposes the tail
    ax = axes[1]
    ax.semilogy(np.arange(len(s_t)), s_t, color=COLORS['teacher'],
                linewidth=2.0, label='Teacher')
    ax.semilogy(np.arange(len(s_s)), s_s, color=COLORS['student'],
                linewidth=2.0, linestyle='--', label='Student')
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Singular value (log)')
    ax.set_title('Spectrum (log scale)')
    ax.legend()

    # Quantitative measure of similarity
    cos_sim = np.dot(s_t, s_s) / (np.linalg.norm(s_t) * np.linalg.norm(s_s))
    rel_diff = np.linalg.norm(s_t - s_s) / np.linalg.norm(s_t)
    fig.suptitle(f'Embedding spectrum: cosine similarity = {cos_sim:.4f}, '
                 f'relative L2 difference = {rel_diff:.4f}',
                 y=1.02, fontsize=11)
    _save(fig, out_path)


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--data_root', default='../data/datasets')
    p.add_argument('--teacher_ckpt', required=True)
    p.add_argument('--student_ckpt', required=True)
    p.add_argument('--max_len', type=int, default=50)
    p.add_argument('--out_dir', default='figures')
    p.add_argument('--n_items_tsne', type=int, default=2000)
    p.add_argument('--n_clusters_tsne', type=int, default=8)
    args_cli = p.parse_args()

    args_ns = _build_args_for_loading(args_cli.dataset, args_cli.data_root,
                                      max_len=args_cli.max_len)
    args_ns.teacher_ckpt = args_cli.teacher_ckpt
    args_ns.student_ckpt = args_cli.student_ckpt

    Path(args_cli.out_dir).mkdir(parents=True, exist_ok=True)
    teacher, student, test_loader, _ = _load_models_and_data(args_ns)

    slug = args_cli.dataset.replace('/', '_')

    plot_tsne_embeddings(
        teacher, student,
        n_items=args_cli.n_items_tsne, n_clusters=args_cli.n_clusters_tsne,
        out_path=os.path.join(args_cli.out_dir, f'tsne_embeddings_{slug}.pdf'),
    )

    plot_denoising_trajectory(
        teacher, student, test_loader, torch.device(args_ns.device),
        out_path=os.path.join(args_cli.out_dir, f'denoising_trajectory_{slug}.pdf'),
    )

    plot_svd_spectrum(
        teacher, student,
        out_path=os.path.join(args_cli.out_dir, f'svd_spectrum_{slug}.pdf'),
    )

    print(f'\nAll three analysis figures written to {args_cli.out_dir}/')


if __name__ == '__main__':
    main()