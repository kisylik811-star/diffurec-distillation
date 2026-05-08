"""
Consistency distillation for DiffuRec, with Reward-Aware extension (RACD)
and ablation switches for Block 2 (design choices).

Implements:
  - ConsistencyDiffuRec: the student diffusion module.
  - ConsistencyStudent:  full student wrapping ConsistencyDiffuRec,
    exposing three loss terms (consistency, cross-entropy, reward).

Block 1 contributes the reward term (RACD); Block 2 contributes three
design switches that justify the configuration of the contributed method:

  parametrization in {'xstart', 'eps'}
      'xstart' -> the network output is interpreted as predicted x_0
                 directly (DiffuRec's original choice).
      'eps'    -> the network output is interpreted as the noise
                 prediction; x_0 is recovered analytically as
                     x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps)
                            / sqrt(alpha_bar_t)
                 The architecture is unchanged. This is the standard
                 DDPM (Ho et al. 2020) parametrization.

  solver         in {'ddim', 'heun'}
      'ddim'   -> single deterministic DDIM step (Euler, 1st order).
                 Cheap, default.
      'heun'   -> 2nd-order Heun-style correction: predict the noise
                 at (x_high, t_high), take a DDIM step to get x_low_pred,
                 predict noise again at (x_low_pred, t_low), average the
                 two noise estimates, redo the DDIM step with the
                 averaged noise. Cost: ~2x teacher forward passes per
                 consistency step. Typically helpful at low NFE
                 (Karras et al. EDM 2022).

  use_ema        in {True, False}
      True     -> consistency target is computed by the EMA-tracked
                 student (canonical Song et al. 2023 setup).
      False    -> target is computed by the online student itself,
                 detached from the graph (no_grad). Equivalent to
                 self-distillation without an EMA anchor.

References:
  Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
  Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
  Song et al., "Consistency Models", ICML 2023.
  Karras et al., "Elucidating the Design Space of Diffusion-Based
                  Generative Models", NeurIPS 2022.
  Qin et al., "A general approximation framework for direct
               optimization of information retrieval measures", IR 2010.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffurec import Diffu_xstart, _extract_into_tensor
from model import LayerNorm


# --------------------------------------------------------------------- #
#  Ranking-reward losses (no learnable parameters)                      #
# --------------------------------------------------------------------- #
def approx_ndcg_loss(scores, target, alpha=10.0, eps=1e-10):
    """
    Differentiable surrogate for -NDCG with a single relevant item per
    query.  Approximate rank of the target:
        rank(t) ~= 1 + sum_{j != t} sigmoid(alpha * (s_j - s_t))
    NDCG = 1 / log2(rank + 1).
    """
    target = target.view(-1, 1)
    target_scores = scores.gather(1, target)
    diff = scores - target_scores
    approx_rank = 1.0 + torch.sigmoid(alpha * diff).sum(dim=-1) - 0.5
    ndcg = 1.0 / torch.log2(approx_rank + 1.0 + eps)
    return -ndcg.mean()


def pairwise_hardneg_margin_loss(scores, target, hard_neg_idx, margin=0.0):
    """Smooth pairwise hinge against teacher-mined hard negatives."""
    target = target.view(-1, 1)
    pos = scores.gather(1, target)
    neg = scores.gather(1, hard_neg_idx)
    return F.softplus(neg - pos + margin).mean()


@torch.no_grad()
def mine_hard_negatives_from_teacher(teacher_scores, target, K=16):
    masked = teacher_scores.clone()
    masked.scatter_(1, target.view(-1, 1), float('-inf'))
    return masked.topk(K, dim=-1).indices


# --------------------------------------------------------------------- #
#  Diffusion-step helpers (parametrization + DDIM/Heun)                 #
# --------------------------------------------------------------------- #
def _eps_to_x0(x_t, t, eps_pred, sqrt_ab, sqrt_one_minus_ab):
    """
    Recover x_0 from a noise prediction:
        x_0 = (x_t - sqrt(1-ab_t) * eps) / sqrt(ab_t)
    """
    sa  = _extract_into_tensor(sqrt_ab,             t, x_t.shape)
    som = _extract_into_tensor(sqrt_one_minus_ab,   t, x_t.shape)
    return (x_t - som * eps_pred) / sa


def _x0_to_eps(x_t, t, x0_pred, sqrt_ab, sqrt_one_minus_ab):
    """Inverse of _eps_to_x0."""
    sa  = _extract_into_tensor(sqrt_ab,           t, x_t.shape)
    som = _extract_into_tensor(sqrt_one_minus_ab, t, x_t.shape)
    return (x_t - sa * x0_pred) / som


def _ddim_step_from_x0(x0, eps, t_low, sqrt_ab, sqrt_one_minus_ab, ref_shape):
    """One deterministic DDIM step: x_t_low = sqrt(ab_low)*x_0 + sqrt(1-ab_low)*eps."""
    sa_l  = _extract_into_tensor(sqrt_ab,             t_low, ref_shape)
    som_l = _extract_into_tensor(sqrt_one_minus_ab,   t_low, ref_shape)
    return sa_l * x0 + som_l * eps


# --------------------------------------------------------------------- #
#  Student diffusion module                                             #
# --------------------------------------------------------------------- #
class ConsistencyDiffuRec(nn.Module):
    """
    Student diffusion module. The same `Diffu_xstart` network is reused;
    its output is interpreted according to `self.parametrization`.
    """

    def __init__(self, teacher_diffurec, args):
        super().__init__()

        # Reuse the teacher's diffusion schedule (frozen, not learned).
        self.betas = teacher_diffurec.betas
        self.alphas_cumprod = teacher_diffurec.alphas_cumprod
        self.alphas_cumprod_prev = teacher_diffurec.alphas_cumprod_prev
        self.sqrt_alphas_cumprod = teacher_diffurec.sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = teacher_diffurec.sqrt_one_minus_alphas_cumprod
        self.num_timesteps = teacher_diffurec.num_timesteps
        self.rescale_timesteps = teacher_diffurec.rescale_timesteps

        # Trainable predictor with the same architecture as the teacher's,
        # initialized from the teacher's weights. The interpretation of
        # the output is governed by `self.parametrization`.
        self.xstart_model = Diffu_xstart(args.hidden_size, args)
        self.xstart_model.load_state_dict(teacher_diffurec.xstart_model.state_dict())

        # Block-2 design switches with safe defaults.
        self.parametrization = str(getattr(args, 'parametrization', 'xstart'))
        self.solver = str(getattr(args, 'solver', 'ddim'))
        if self.parametrization not in {'xstart', 'eps'}:
            raise ValueError(f'parametrization must be xstart or eps, got {self.parametrization}')
        if self.solver not in {'ddim', 'heun'}:
            raise ValueError(f'solver must be ddim or heun, got {self.solver}')

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_sample(self, x_start, t, noise):
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ---- Network calls in the chosen parametrization ---- #
    def _net_predict_x0_and_eps(self, item_rep, x_t, t, mask_seq, network=None):
        """
        Run the network once and return BOTH x_0 and eps consistent with each
        other.  `network` defaults to self.xstart_model; pass an EMA copy or a
        teacher network to score with those.

        For 'xstart': network output is x_0; eps derived analytically.
        For 'eps':    network output is eps; x_0 derived analytically.
        """
        net = network if network is not None else self.xstart_model
        out, _ = net(item_rep, x_t, self._scale_timesteps(t), mask_seq)

        if self.parametrization == 'xstart':
            x0 = out
            eps = _x0_to_eps(x_t, t, x0,
                             self.sqrt_alphas_cumprod,
                             self.sqrt_one_minus_alphas_cumprod)
        else:  # eps
            eps = out
            x0 = _eps_to_x0(x_t, t, eps,
                            self.sqrt_alphas_cumprod,
                            self.sqrt_one_minus_alphas_cumprod)
        return x0, eps

    def predict_x0(self, item_rep, x_t, t, mask_seq, network=None):
        """Consistency function f_theta(x_t, t, history) -> predicted x_0."""
        x0, _ = self._net_predict_x0_and_eps(item_rep, x_t, t, mask_seq, network=network)
        return x0

    # ---- Teacher-driven solver step (DDIM or Heun) ---- #
    @torch.no_grad()
    def teacher_solver_step(self, teacher_diffu, item_rep,
                            x_t_high, t_high, t_low, mask_seq):
        """
        One step of the teacher-induced ODE from t_high down to t_low.
        Always returns (x_t_low, x0_at_t_high). The latter is exposed for
        downstream uses (e.g. hard-negative mining).

        Solvers:
          - 'ddim': Euler-style, one teacher forward pass.
                       eps_h = network_eps(x_t_high, t_high)
                       x_t_low = sqrt(ab_low)*x0_h + sqrt(1-ab_low)*eps_h

          - 'heun': 2nd-order correction, two teacher forward passes.
                       (x_h, eps_h)              at t_high
                       x_pred_low = ddim_step(x_h, eps_h, t_low)
                       (_, eps_l)                at t_low using x_pred_low
                       eps_avg   = 0.5 * (eps_h + eps_l)
                       x_t_low   = sqrt(ab_low)*x0_h + sqrt(1-ab_low)*eps_avg
        """
        # First teacher call at the high end.
        x0_h, eps_h = self._net_predict_x0_and_eps(
            item_rep, x_t_high, t_high, mask_seq, network=teacher_diffu.xstart_model
        )

        if self.solver == 'ddim':
            x_t_low = _ddim_step_from_x0(
                x0_h, eps_h, t_low,
                self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod,
                x_t_high.shape,
            )
            return x_t_low, x0_h

        # Heun: do an Euler predict, then correct with eps at the low end.
        x_pred_low = _ddim_step_from_x0(
            x0_h, eps_h, t_low,
            self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod,
            x_t_high.shape,
        )
        _, eps_l = self._net_predict_x0_and_eps(
            item_rep, x_pred_low, t_low, mask_seq, network=teacher_diffu.xstart_model
        )
        eps_avg = 0.5 * (eps_h + eps_l)
        x_t_low = _ddim_step_from_x0(
            x0_h, eps_avg, t_low,
            self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod,
            x_t_high.shape,
        )
        return x_t_low, x0_h

    # ---- Inference: alternate denoise / re-noise (Algorithm 1) ---- #
    @torch.no_grad()
    def sample(self, item_rep, mask_seq, num_steps=1):
        """Generate x_0 with the student in `num_steps` forward passes."""
        device = next(self.parameters()).device
        bs = item_rep.shape[0]
        H = item_rep.shape[-1]
        T = self.num_timesteps

        x_t = torch.randn(bs, H, device=device)

        if num_steps == 1:
            t = torch.full((bs,), T - 1, device=device, dtype=torch.long)
            return self.predict_x0(item_rep, x_t, t, mask_seq)

        ts = np.linspace(T - 1, 1, num_steps).round().astype(int)
        x_0 = None
        for i, t_val in enumerate(ts):
            t = torch.full((bs,), int(t_val), device=device, dtype=torch.long)
            x_0 = self.predict_x0(item_rep, x_t, t, mask_seq)
            if i < len(ts) - 1:
                noise = torch.randn_like(x_0)
                t_next = torch.full((bs,), int(ts[i + 1]), device=device, dtype=torch.long)
                x_t = self.q_sample(x_0, t_next, noise)
        return x_0


# --------------------------------------------------------------------- #
#  Full student model                                                   #
# --------------------------------------------------------------------- #
class ConsistencyStudent(nn.Module):
    """
    Full student: item embeddings + LayerNorm + ConsistencyDiffuRec.

    `consistency_loss` returns (cons_loss, ce_loss, reward_loss).

    Block-2 switches honoured here:
      - args.parametrization (handled inside ConsistencyDiffuRec)
      - args.solver           (handled inside ConsistencyDiffuRec)
      - args.use_ema          (handled here: gates EMA target + EMA updates)
    """

    def __init__(self, teacher_model, args, ema_decay=0.95):
        super().__init__()
        self.args = args
        self.ema_decay = ema_decay
        self.use_ema = bool(getattr(args, 'use_ema', True))

        self.emb_dim = args.hidden_size
        self.item_num = args.item_num + 1

        # Initialize embeddings/norm from the teacher.
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.item_embeddings.load_state_dict(teacher_model.item_embeddings.state_dict())
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.LayerNorm.load_state_dict(teacher_model.LayerNorm.state_dict())
        self.dropout = nn.Dropout(args.dropout)

        # Online network and (optional) EMA target network.
        self.diffu_student = ConsistencyDiffuRec(teacher_model.diffu, args)
        if self.use_ema:
            self.diffu_student_ema = copy.deepcopy(self.diffu_student)
            for p in self.diffu_student_ema.parameters():
                p.requires_grad = False
        else:
            self.diffu_student_ema = None  # explicit; no parameters to track

        self.loss_ce = nn.CrossEntropyLoss()

        # Reward-config defaults (Block 1).
        self.ndcg_weight   = float(getattr(args, 'ndcg_weight', 0.0))
        self.margin_weight = float(getattr(args, 'margin_weight', 0.0))
        self.ndcg_alpha    = float(getattr(args, 'ndcg_alpha', 10.0))
        self.hard_neg_k    = int(getattr(args, 'hard_neg_k', 16))
        self.margin_value  = float(getattr(args, 'margin_value', 0.0))

    @torch.no_grad()
    def update_ema(self):
        """No-op when use_ema is False; ordinary Polyak averaging otherwise."""
        if not self.use_ema:
            return
        for p_t, p_o in zip(self.diffu_student_ema.parameters(),
                            self.diffu_student.parameters()):
            p_t.data.mul_(self.ema_decay).add_(p_o.data, alpha=1 - self.ema_decay)

    def encode(self, sequence):
        e = self.item_embeddings(sequence)
        e = self.embed_dropout(e)
        e = self.LayerNorm(e)
        mask = (sequence > 0).float()
        return e, mask

    def _scores(self, pred_x0):
        return torch.matmul(pred_x0, self.item_embeddings.weight.t())

    def _reward_loss(self, scores, target, teacher_scores=None):
        device = scores.device
        total = torch.zeros((), device=device)
        if self.ndcg_weight > 0.0:
            total = total + self.ndcg_weight * approx_ndcg_loss(
                scores, target, alpha=self.ndcg_alpha
            )
        if self.margin_weight > 0.0:
            assert teacher_scores is not None, (
                'margin_weight > 0 but teacher_scores were not provided'
            )
            hard_neg = mine_hard_negatives_from_teacher(
                teacher_scores, target, K=self.hard_neg_k
            )
            total = total + self.margin_weight * pairwise_hardneg_margin_loss(
                scores, target, hard_neg, margin=self.margin_value
            )
        return total

    def consistency_loss(self, sequence, target, teacher_diffu, teacher_model=None):
        """
        Returns (cons_loss, ce_loss, reward_loss).

        cons_loss target:
          - if use_ema: EMA-tracked student at (x_t_low, t_low)
          - else:       online student at (x_t_low, t_low), no-grad
        """
        item_rep, mask_seq = self.encode(sequence)
        x_0 = self.item_embeddings(target.squeeze(-1))  # B x H

        bs = sequence.size(0)
        T = self.diffu_student.num_timesteps
        device = sequence.device

        n = torch.randint(1, T, (bs,), device=device)
        t_high = n
        t_low = n - 1

        noise = torch.randn_like(x_0)
        x_t_high = self.diffu_student.q_sample(x_0, t_high, noise)

        # Teacher-driven step (DDIM or Heun, see ConsistencyDiffuRec.solver).
        with torch.no_grad():
            x_t_low, teacher_x0 = self.diffu_student.teacher_solver_step(
                teacher_diffu, item_rep, x_t_high, t_high, t_low, mask_seq
            )

        # Online prediction at the high end (gradient flows here).
        pred_high = self.diffu_student.predict_x0(item_rep, x_t_high, t_high, mask_seq)

        # Target prediction at the low end (no gradient).
        with torch.no_grad():
            if self.use_ema:
                pred_low = self.diffu_student_ema.predict_x0(
                    item_rep, x_t_low, t_low, mask_seq
                )
            else:
                # Self-distillation without EMA anchor: use the online student
                # itself but stop gradients via no_grad (the surrounding context
                # already does this).
                pred_low = self.diffu_student.predict_x0(
                    item_rep, x_t_low, t_low, mask_seq
                )

        cons_loss = F.mse_loss(pred_high, pred_low)

        scores = self._scores(pred_high)
        ce_loss = self.loss_ce(scores, target.squeeze(-1))

        teacher_scores = None
        if self.margin_weight > 0.0:
            with torch.no_grad():
                if teacher_model is not None:
                    teacher_scores = torch.matmul(
                        teacher_x0, teacher_model.item_embeddings.weight.t()
                    )
                else:
                    teacher_scores = self._scores(teacher_x0)

        reward_loss = self._reward_loss(scores, target, teacher_scores=teacher_scores)
        return cons_loss, ce_loss, reward_loss

    @torch.no_grad()
    def predict_scores(self, sequence, num_steps=1):
        item_rep, mask_seq = self.encode(sequence)
        x_0 = self.diffu_student.sample(item_rep, mask_seq, num_steps=num_steps)
        return self._scores(x_0)