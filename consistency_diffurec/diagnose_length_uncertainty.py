import torch
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def compute_teacher_uncertainty(teacher, test_loader, device, n_samples=10):
    """
    Для каждого батча считаем variance предсказаний teacher'а
    при разных сэмплах начального шума.
    """
    teacher.eval()
    length_uncertainty = {}  # length -> list of variances
    
    for batch in tqdm(test_loader):
        seq, target = [x.to(device) for x in batch]
        seq_len = (seq > 0).sum(dim=-1).cpu().numpy()
        
        # Кодируем историю один раз
        item_emb = teacher.item_embeddings(seq)
        item_emb = teacher.embed_dropout(item_emb)
        item_emb = teacher.LayerNorm(item_emb)
        mask_seq = (seq > 0).float()
        
        bs, H = item_emb.shape[0], item_emb.shape[-1]
        
        # Несколько запусков с разным шумом
        all_preds = []
        for _ in range(n_samples):
            x_t = torch.randn(bs, H, device=device)
            rep_diffu, _, _, _, _, _ = teacher.diffu(item_emb, x_t, mask_seq, train_flag=False)
            scores = teacher.diffu_rep_pre(rep_diffu)
            pred_item = scores.argmax(dim=-1)
            all_preds.append(pred_item.cpu())
        
        # Считаем, насколько предсказания различаются
        all_preds = torch.stack(all_preds)  # n_samples x B
        for i, length in enumerate(seq_len):
            predictions_for_sample = all_preds[:, i]
            # Uncertainty = доля запусков, где предсказание не совпадает с модой
            mode = torch.mode(predictions_for_sample)[0]
            uncertainty = (predictions_for_sample != mode).float().mean().item()
            
            length = int(length)
            if length not in length_uncertainty:
                length_uncertainty[length] = []
            length_uncertainty[length].append(uncertainty)
    
    # Агрегируем по квартилям длины
    lengths = sorted(length_uncertainty.keys())
    quartiles = {
        'short': [],   # bottom 25%
        'medium': [],  # middle 50%
        'long': []     # top 25%
    }
    
    for l in lengths:
        if l < np.percentile(lengths, 25):
            quartiles['short'].extend(length_uncertainty[l])
        elif l > np.percentile(lengths, 75):
            quartiles['long'].extend(length_uncertainty[l])
        else:
            quartiles['medium'].extend(length_uncertainty[l])
    
    for name, vals in quartiles.items():
        print(f"{name}: mean uncertainty = {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    return quartiles