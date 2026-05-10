import os
import argparse
import pickle
import torch
import numpy as np
from collections import defaultdict
from model import createmodel_diffu, AttDiffusemodel
from utils import DataTrain, DataVal, DataTest

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='toys', help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--emb_dropout', type=float, default=0.3)
    parser.add_argument('--hidden_act', default='gelu', type=str)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--schedule_sampler_name', type=str, default='lossaware')
    parser.add_argument('--diffusion_steps', type=int, default=32)
    parser.add_argument('--lambda_uncertainty', type=float, default=0.001)
    parser.add_argument('--noise_schedule', default='trunclin')
    parser.add_argument('--rescale_timesteps', default=True)
    return parser.parse_args()

def load_data_and_model(args):
    with open(args.data_path, 'rb') as f:
        data_raw = pickle.load(f)
    
    # Считаем количество айтемов
    item_list = []
    for seq in data_raw['train'].values():
        item_list.extend(seq)
    args.itemnum = max(item_list) + 1
    
    print(f"Dataset loaded. Item num: {args.itemnum}")
    
    # Создаем модель
    diffu_pre = createmodel_diffu(args)
    teacher = AttDiffusemodel(diffu_pre, args)
    
    # Загружаем веса
    state_dict = torch.load(args.teacher_path, map_location=args.device)
    teacher.load_state_dict(state_dict)
    teacher.to(args.device)
    teacher.eval()
    print("Teacher model loaded successfully.")
    
    return data_raw, teacher

def get_test_batch(data_raw, args, num_samples=10):
    seqs = []
    lengths = []
    keys = list(data_raw['train'].keys())[:num_samples]
    
    for k in keys:
        seq = data_raw['train'][k]
        lengths.append(min(len(seq), args.maxlen))
        seq = seq[-args.maxlen:]
        padding_len = args.maxlen - len(seq)
        seq = [0] * padding_len + seq
        seqs.append(seq)
        
    seqs = torch.LongTensor(seqs).to(args.device)
    mask_seq = (seqs != 0).float()
    return seqs, mask_seq, lengths

def run_diagnostics():
    args = parse_args()
    data_raw, teacher = load_data_and_model(args)
    
    seqs, mask_seq, lengths = get_test_batch(data_raw, args, num_samples=20)
    item_embeddings = teacher.item_embeddings(seqs)
    
    print("\n" + "="*50)
    print("🚀 НАЧАЛО ДИАГНОСТИЧЕСКИХ ПРОВЕРОК")
    print("="*50)
    
    # Проверка 1: Стохастичность xstart_model (разный шум)
    print("\n[Проверка 1] Стохастичность предсказаний (Item IDs)")
    t_step = torch.full((seqs.shape[0],), args.diffusion_steps - 1, device=args.device)
    preds_list = []
    for _ in range(5):
        noise_xt = torch.randn_like(item_embeddings)
        with torch.no_grad():
            x0_pred, _ = teacher.diffu.xstart_model(item_embeddings, noise_xt, t_step, mask_seq)
            scores = torch.matmul(x0_pred, teacher.item_embeddings.weight.t())
            preds = scores.argmax(dim=-1)[:, -1] # берем последний айтем
            preds_list.append(preds.cpu().numpy())
    
    preds_stack = np.stack(preds_list, axis=0)
    is_stochastic = not np.all(preds_stack == preds_stack[0, :])
    print(f"Меняется ли предсказанный ID при разном шуме? {'✅ ДА' if is_stochastic else '❌ НЕТ (Детерминировано)'}")
    print(f"Пример предсказаний для пользователя 0: {preds_stack[:, 0]}")

    # Проверка 2: Зависимость variance от timestep t
    print("\n[Проверка 2] Зависимость variance от timestep t")
    timesteps_to_check = [1, 10, 20, 31]
    for t_val in timesteps_to_check:
        t_step = torch.full((seqs.shape[0],), t_val, device=args.device)
        x0_embs = []
        for _ in range(10):
            noise_xt = torch.randn_like(item_embeddings)
            with torch.no_grad():
                x0_pred, _ = teacher.diffu.xstart_model(item_embeddings, noise_xt, t_step, mask_seq)
                x0_embs.append(x0_pred.cpu().numpy())
        var_t = np.var(np.stack(x0_embs), axis=0).mean()
        print(f"Variance при t={t_val:2d}: {var_t:.6f}")
    
    # Проверка 3: Variance в embedding space (t=31)
    print("\n[Проверка 3] Variance в embedding space")
    t_step = torch.full((seqs.shape[0],), 31, device=args.device)
    x0_embs = []
    for _ in range(10):
        noise_xt = torch.randn_like(item_embeddings)
        with torch.no_grad():
            x0_pred, _ = teacher.diffu.xstart_model(item_embeddings, noise_xt, t_step, mask_seq)
            x0_embs.append(x0_pred.cpu().numpy())
    var_emb = np.var(np.stack(x0_embs), axis=0).mean()
    print(f"Средняя variance x_0: {var_emb:.6f} (Если > 0, предсказания стохастичны)")

    # Проверка 4: Влияет ли train_flag (Dropout & Uncertainty)
    print("\n[Проверка 4] Влияет ли train_flag на variance?")
    teacher.train()
    x0_embs_train = []
    for _ in range(10):
        noise_xt = torch.randn_like(item_embeddings)
        with torch.no_grad():
            x0_pred, _ = teacher.diffu.xstart_model(item_embeddings, noise_xt, t_step, mask_seq)
            x0_embs_train.append(x0_pred.cpu().numpy())
    var_train = np.var(np.stack(x0_embs_train), axis=0).mean()
    teacher.eval()
    print(f"Variance в режиме .eval():  {var_emb:.6f}")
    print(f"Variance в режиме .train(): {var_train:.6f}")
    
    # Проверка 6: Length-Aware Uncertainty (ДЛЯ LACD)
    print("\n[Проверка 6] Гипотеза LACD: Зависит ли variance от длины истории?")
    len_variance = defaultdict(list)
    x0_embs_stack = np.stack(x0_embs) # shape: (10, batch, seq_len, hidden)
    vars_per_user = np.var(x0_embs_stack, axis=0).mean(axis=(1, 2)) # усредняем по sequence и hidden
    
    for length, var in zip(lengths, vars_per_user):
        # Группируем по корзинам длин (0-10, 11-20, etc.)
        bucket = (length // 10) * 10
        len_variance[f"{bucket}-{bucket+9}"].append(var)
        
    for bucket in sorted(len_variance.keys()):
        avg_var = np.mean(len_variance[bucket])
        print(f"Длина истории {bucket:^7} -> Средняя variance: {avg_var:.6f}")
    print("💡 Вывод: Если короткие истории имеют БОЛЬШУЮ variance, значит учитель в них менее уверен, и LACD точно нужен!")

    # Проверка 7: Position-Aware Uncertainty
    print("\n[Проверка 7] Гипотеза Position-Aware: Как меняется variance по позициям?")
    # Усредняем по батчу (игнорируя паддинги) и по фичам
    # mask_seq shape: (batch, seq_len)
    var_by_pos = np.var(x0_embs_stack, axis=0).mean(axis=-1) # (batch, seq_len)
    
    # Считаем среднюю variance для последних 5 позиций и первых 5 реальных позиций
    # Для простоты выведем среднее по позициям (без учета паддинга)
    valid_vars = var_by_pos * mask_seq.cpu().numpy()
    avg_var_by_pos = valid_vars.sum(axis=0) / (mask_seq.cpu().numpy().sum(axis=0) + 1e-9)
    
    # Выводим с конца (позиция -1 это последний айтем)
    print(f"Variance на позиции (t-5): {avg_var_by_pos[-5]:.6f}")
    print(f"Variance на позиции (t-4): {avg_var_by_pos[-4]:.6f}")
    print(f"Variance на позиции (t-3): {avg_var_by_pos[-3]:.6f}")
    print(f"Variance на позиции (t-2): {avg_var_by_pos[-2]:.6f}")
    print(f"Variance на позиции (t-1): {avg_var_by_pos[-1]:.6f} (Последний таргет)")
    print("💡 Вывод: Если ближе к концу variance падает или растет — это сильный аргумент для Position-aware weighting.")

if __name__ == '__main__':
    run_diagnostics()