"""
Consistency distillation for DiffuRec.

Implements:
  - ConsistencyDiffuRec: the student diffusion module that maps any
    (x_t, t, history) to predicted x_0 in a single forward pass.
  - ConsistencyStudent: full student model wrapping ConsistencyDiffuRec
    with the same item-embedding / LayerNorm setup as the teacher
    Att_Diffuse_model.

DiffuRec uses discrete-time DDPM with T = num_timesteps reverse steps.
The teacher's one-step ODE estimate (used to build consistency targets)
is implemented as a deterministic DDIM step.

Reference: Song et al., "Consistency Models", ICML 2023.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffurec import Diffu_xstart, _extract_into_tensor
from model import LayerNorm


class ConsistencyDiffuRec(nn.Module):
    """Student diffusion module trained via consistency distillation."""

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
        # initialized from the teacher's weights.
        self.xstart_model = Diffu_xstart(args.hidden_size, args)
        self.xstart_model.load_state_dict(teacher_diffurec.xstart_model.state_dict())

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_sample(self, x_start, t, noise):
        """x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise."""
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_x0(self, item_rep, x_t, t, mask_seq):
        """Consistency function f_theta(x_t, t, history) -> predicted x_0."""
        x_0_pred, _ = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)
        return x_0_pred

    @torch.no_grad()
    def teacher_ddim_step(self, teacher, item_rep, x_t_high, t_high, t_low, mask_seq):
        """Deterministic DDIM step from t_high down to t_low using the frozen teacher."""
        x_0_t, _ = teacher.xstart_model(
            item_rep, x_t_high, teacher._scale_timesteps(t_high), mask_seq
        )

        sa_h = _extract_into_tensor(teacher.sqrt_alphas_cumprod, t_high, x_t_high.shape)
        som_h = _extract_into_tensor(teacher.sqrt_one_minus_alphas_cumprod, t_high, x_t_high.shape)
        sa_l = _extract_into_tensor(teacher.sqrt_alphas_cumprod, t_low, x_t_high.shape)
        som_l = _extract_into_tensor(teacher.sqrt_one_minus_alphas_cumprod, t_low, x_t_high.shape)

        eps_pred = (x_t_high - sa_h * x_0_t) / som_h
        x_t_low = sa_l * x_0_t + som_l * eps_pred
        return x_t_low

    @torch.no_grad()
    def sample(self, item_rep, mask_seq, num_steps=1):
        """
        Generate x_0 with the student.

        num_steps = 1: a single forward pass (canonical use of consistency models).
        num_steps > 1: alternate denoise / re-noise (Algorithm 1 from the paper).
        """
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


class ConsistencyStudent(nn.Module):
    """
    Full student model: item embeddings + LayerNorm + ConsistencyDiffuRec.
    Mirrors Att_Diffuse_model but with a consistency-trained diffusion module.
    """

    def __init__(self, teacher_model, args, ema_decay=0.95):
        super().__init__()
        self.args = args
        self.ema_decay = ema_decay

        self.emb_dim = args.hidden_size
        self.item_num = args.item_num + 1

        # Initialize embeddings/norm from the teacher.
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.item_embeddings.load_state_dict(teacher_model.item_embeddings.state_dict())
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.LayerNorm.load_state_dict(teacher_model.LayerNorm.state_dict())
        self.dropout = nn.Dropout(args.dropout)

        # Online network and EMA target network.
        self.diffu_student = ConsistencyDiffuRec(teacher_model.diffu, args)
        self.diffu_student_ema = copy.deepcopy(self.diffu_student)
        for p in self.diffu_student_ema.parameters():
            p.requires_grad = False

        self.loss_ce = nn.CrossEntropyLoss()

    @torch.no_grad()
    def update_ema(self):
        for p_t, p_o in zip(self.diffu_student_ema.parameters(),
                            self.diffu_student.parameters()):
            p_t.data.mul_(self.ema_decay).add_(p_o.data, alpha=1 - self.ema_decay)

    def encode(self, sequence):
        e = self.item_embeddings(sequence)
        e = self.embed_dropout(e)
        e = self.LayerNorm(e)
        mask = (sequence > 0).float()
        return e, mask

    def consistency_loss(self, sequence, target, teacher_diffu):
        """
        Returns (consistency_loss, ce_loss).

        consistency_loss enforces self-consistency along the teacher-induced
        ODE trajectory; ce_loss keeps the student grounded in the recsys task.
        """
        item_rep, mask_seq = self.encode(sequence)
        x_0 = self.item_embeddings(target.squeeze(-1))  # B x H

        bs = sequence.size(0)
        T = self.diffu_student.num_timesteps
        device = sequence.device

        # Sample n in [1, T-1]; pair (t_high, t_low) = (n, n-1) are adjacent.
        n = torch.randint(1, T, (bs,), device=device)
        t_high = n
        t_low = n - 1

        noise = torch.randn_like(x_0)
        x_t_high = self.diffu_student.q_sample(x_0, t_high, noise)

        # Consistency target: one DDIM step from t_high down to t_low using
        # the frozen teacher.
        with torch.no_grad():
            x_t_low = self.diffu_student.teacher_ddim_step(
                teacher_diffu, item_rep, x_t_high, t_high, t_low, mask_seq
            )

        pred_high = self.diffu_student.predict_x0(item_rep, x_t_high, t_high, mask_seq)
        with torch.no_grad():
            pred_low = self.diffu_student_ema.predict_x0(item_rep, x_t_low, t_low, mask_seq)

        cons_loss = F.mse_loss(pred_high, pred_low)

        scores = torch.matmul(pred_high, self.item_embeddings.weight.t())
        ce_loss = self.loss_ce(scores, target.squeeze(-1))

        return cons_loss, ce_loss

    @torch.no_grad()
    def predict_scores(self, sequence, num_steps=1):
        """Inference: scores over all items in num_steps NFEs."""
        item_rep, mask_seq = self.encode(sequence)
        x_0 = self.diffu_student.sample(item_rep, mask_seq, num_steps=num_steps)
        return torch.matmul(x_0, self.item_embeddings.weight.t())