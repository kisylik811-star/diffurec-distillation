"""
Быстрая диагностика variance teacher'а для DiffuRec.
Запуск в Colab:
    !python check_teacher_variance.py
"""

import os
import sys
import pickle
import torch
import numpy as np
from collections import defaultdict

# Добавляем пути к исходникам оригинального DiffuRec
sys.path.insert(0, '/content/diffurec-distillation/original_diffurec')
sys.path.insert(0, '/content/diffurec-distillation/consistency_diffurec')

from model import create_model_diffu, Att_Diffuse_model
from utils import Data_Test


def create_args():
    """Создаёт args для загрузки teacher модели."""
    class Args:
        pass
    
    args = Args()
    
    # Dataset params
    args.dataset = 'toys'
    args.data_root = '/content/diffurec-distillation/original_diffurec/../datasets/data'
    args.max_len = 50
    
    # Model params
    args.hidden_size = 128
    args.dropout = 0.1
    args.emb_dropout = 0.3
    args.hidden_act = 'gelu'
    args.num_blocks = 4
    args.diffusion_steps = 32
    args.lambda_uncertainty = 0.001
    args.noise_schedule = 'trunc_lin'
    args.schedule_sampler_name = 'lossaware'
    args.rescale_timesteps = True
    
    # Other
    args.long_head = False
    args.diversity_measure = False
    args.epoch_time_avg = False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.batch_size = 256
    
    return args


def load_data(args):
    """Загружает данные и возвращает test_loader."""
    data_path = os.path.join(args.data_root, args.dataset, 'dataset.pkl')
    
    if not os.path.exists(data_path):
        # Пробуем альтернативные пути
        alt_paths = [
            '/content/diffurec-distillation/datasets/data/toys/dataset.pkl',
            '/content/datasets/data/toys/dataset.pkl',
            '../datasets/data/toys/dataset.pkl',
        ]
        for p in alt_paths:
            if os.path.exists(p):
                data_path = p
                break
    
    print(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        data_raw = pickle.load(f)
    
    args.item_num = len(data_raw['smap'])
    
    test_loader = Data_Test(
        data_raw['train'], data_raw['val'], data_raw['test'], args
    ).get_pytorch_dataloaders()
    
    return test_loader, data_raw


def load_teacher(args, ckpt_path):
    """Загружает teacher модель."""
    teacher = Att_Diffuse_model(create_model_diffu(args), args).to(args.device)
    
    if os.path.exists(ckpt_path):
        print(f"Loading teacher from {ckpt_path}")
        teacher.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    else:
        # Пробуем альтернативные пути
        alt_paths = [
            '/content/drive/MyDrive/diffurec-distillation/consistency_diffurec/checkpoints/teacher_toys.pt',
            'checkpoints/teacher_toys.pt',
            '/content/diffurec-distillation/consistency_diffurec/checkpoints/teacher_toys.pt',
        ]
        for p in alt_paths:
            if os.path.exists(p):
                print(f"Loading teacher from {p}")
                teacher.load_state_dict(torch.load(p, map_location=args.device))
                break
        else:
            raise FileNotFoundError(f"Teacher checkpoint not found in any location")
    
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    return teacher


@torch.no_grad()
def check_teacher_variance(teacher, test_loader, device, n_samples=20):
    """
    Проверяет variance предсказаний teacher'а.
    
    Args:
        teacher: модель
        test_loader: DataLoader
        device: cuda/cpu
        n_samples: количество сэмплов шума для каждого примера
    
    Returns:
        variance_mean: средняя variance в embedding space
        uncertainties_by_length: dict {length: list[uncertainties]}
    """
    teacher.eval()
    
    # Берём первый батч для быстрой диагностики
    batch = next(iter(test_loader))
    seq, target = [x.to(device) for x in batch]
    
    # Ограничиваем размер для скорости
    max_samples = 32
    if seq.size(0) > max_samples:
        seq = seq[:max_samples]
        target = target[:max_samples]
    
    bs = seq.size(0)
    print(f"\nChecking {bs} sequences with {n_samples} noise samples each...")
    
    # Кодируем историю один раз (одинаково для всех сэмплов)
    item_emb = teacher.item_embeddings(seq)
    item_emb = teacher.embed_dropout(item_emb)
    item_emb = teacher.LayerNorm(item_emb)
    mask_seq = (seq > 0).float()
    seq_lengths = mask_seq.sum(dim=-1).cpu().numpy().astype(int)
    
    H = item_emb.size(-1)
    
    # Массивы для сбора результатов
    all_pred_items = []  # n_samples x B
    all_x0 = []          # n_samples x B x H
    
    for sample_i in range(n_samples):
        # Новый начальный шум для каждого сэмпла
        x_t = torch.randn(bs, H, device=device)
        
        # Forward через diffusion
        rep_diffu, _, _, _, _, _ = teacher.diffu(
            item_emb, x_t, mask_seq, train_flag=False
        )
        all_x0.append(rep_diffu.cpu())
        
        # Получаем предсказанный item
        scores = teacher.diffu_rep_pre(rep_diffu)
        pred_items = scores.argmax(dim=-1).cpu()
        all_pred_items.append(pred_items)
    
    # Преобразуем в тензоры
    all_pred_items = torch.stack(all_pred_items)  # n_samples x B
    all_x0 = torch.stack(all_x0)                  # n_samples x B x H
    
    # --- Анализ ---
    uncertainties_by_length = defaultdict(list)
    
    print("\n" + "="*70)
    print("VARIANCE ANALYSIS BY SEQUENCE LENGTH")
    print("="*70)
    print(f"{'Sample':<8} {'Length':<8} {'Unique preds':<14} {'Mode freq':<12} {'Uncertainty':<12}")
    print("-"*70)
    
    for i in range(bs):
        preds_for_item = all_pred_items[:, i]  # n_samples
        unique_preds = torch.unique(preds_for_item)
        n_unique = len(unique_preds)
        
        # Находим моду (самое частое предсказание)
        mode_pred = torch.mode(preds_for_item)[0].item()
        mode_count = (preds_for_item == mode_pred).sum().item()
        uncertainty = 1.0 - (mode_count / n_samples)
        
        length = seq_lengths[i]
        uncertainties_by_length[length].append(uncertainty)
        
        # Печатаем только интересные случаи (где uncertainty > 0)
        marker = "⚠️" if uncertainty > 0 else "  "
        print(f"{marker} {i:<6} {length:<8} {n_unique:<14} {mode_count}/{n_samples:<9} {uncertainty:.4f}")
        
        if n_unique > 1 and n_unique <= 5:
            print(f"         -> Predictions: {preds_for_item.tolist()}")
    
    # Средняя variance в embedding space
    var_in_embedding = all_x0.var(dim=0).mean().item()
    
    # Статистика по квартилям длины
    print("\n" + "="*70)
    print("STATISTICS BY LENGTH QUARTILES")
    print("="*70)
    
    lengths = sorted(uncertainties_by_length.keys())
    all_uncertainties = []
    all_lengths = []
    
    for l, uncs in uncertainties_by_length.items():
        all_uncertainties.extend(uncs)
        all_lengths.extend([l] * len(uncs))
    
    if len(all_lengths) > 0:
        len_25 = np.percentile(all_lengths, 25)
        len_75 = np.percentile(all_lengths, 75)
        
        short_unc = []
        medium_unc = []
        long_unc = []
        
        for l, uncs in uncertainties_by_length.items():
            if l <= len_25:
                short_unc.extend(uncs)
            elif l <= len_75:
                medium_unc.extend(uncs)
            else:
                long_unc.extend(uncs)
        
        print(f"Short histories (≤{int(len_25)} items):  mean={np.mean(short_unc):.6f} ± {np.std(short_unc):.6f} (n={len(short_unc)})")
        print(f"Medium histories ({int(len_25)+1}-{int(len_75)}): mean={np.mean(medium_unc):.6f} ± {np.std(medium_unc):.6f} (n={len(medium_unc)})")
        print(f"Long histories (≥{int(len_75)+1} items):   mean={np.mean(long_unc):.6f} ± {np.std(long_unc):.6f} (n={len(long_unc)})")
        
        diff = np.mean(long_unc) - np.mean(short_unc)
        print(f"\nDifference (long - short): {diff:+.6f}")
        
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        if diff < -0.05:
            print("✅ STRONG SIGNAL: Long histories have LOWER uncertainty")
            print("   → LACD with adaptive gap is RECOMMENDED")
        elif diff < -0.01:
            print("✓ WEAK SIGNAL: Long histories have slightly lower uncertainty")
            print("   → LACD may help, but Position+Trajectory is safer")
        else:
            print("✗ NO SIGNAL: Uncertainty does not depend on sequence length")
            print("   → DO NOT use LACD")
            print("   → Use Position+Trajectory weighting (guaranteed to work)")
    
    print(f"\nMean variance in embedding space: {var_in_embedding:.6f}")
    print("="*70)
    
    return var_in_embedding, dict(uncertainties_by_length)


def main():
    print("="*70)
    print("TEACHER VARIANCE DIAGNOSTICS FOR DiffuRec")
    print("="*70)
    
    # Создаём args и загружаем данные
    args = create_args()
    print(f"Device: {args.device}")
    
    # Загружаем данные
    test_loader, data_raw = load_data(args)
    print(f"Item vocab size: {args.item_num}")
    
    # Загружаем teacher
    ckpt_path = '/content/drive/MyDrive/diffurec-distillation/consistency_diffurec/checkpoints/teacher_toys.pt'
    teacher = load_teacher(args, ckpt_path)
    
    # Запускаем диагностику
    variance, uncertainties_by_length = check_teacher_variance(
        teacher, test_loader, args.device, n_samples=20
    )
    
    print("\n✅ Diagnostics complete!")


if __name__ == '__main__':
    main()