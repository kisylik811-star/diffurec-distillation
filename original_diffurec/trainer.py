import torch.nn as nn
import os
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import time
import pickle
import glob


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  


def LSHT_inference(model_joint, args, data_loader):
    device = args.device
    model_joint = model_joint.to(device)
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]
            
            scores_rec, rep_diffu, _, _, _, _ = model_joint(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu)
            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1], [5, 10, 20])
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print(test_metrics_dict_mean)


def save_checkpoint(model, optimizer, epoch, checkpoint_dir, args):
    checkpoint_path = os.path.join(checkpoint_dir, f'teacher_epoch_{epoch}.pt')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'[Checkpoint] saved to {checkpoint_path}')
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f'[Checkpoint] loaded from {checkpoint_path} (epoch {epoch})')
    return epoch


def find_latest_checkpoint(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'teacher_epoch_*.pt'))
    if not checkpoints:
        return None, 0
    
    epochs_found = []
    for c in checkpoints:
        try:
            epoch_num = int(c.split('_epoch_')[1].split('.')[0])
            epochs_found.append((epoch_num, c))
        except:
            pass
    
    if not epochs_found:
        return None, 0
    
    max_epoch, latest_path = max(epochs_found, key=lambda x: x[0])
    return latest_path, max_epoch


def model_train(tra_data_loader, val_data_loader, test_data_loader, model_joint, args, logger):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model_joint = model_joint.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model_joint = nn.DataParallel(model_joint)
    optimizer = optimizers(model_joint, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    
    start_epoch = 0
    checkpoint_dir = getattr(args, 'teacher_checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    latest_checkpoint, latest_epoch = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                model_joint.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                for _ in range(start_epoch):
                    lr_scheduler.step()
                print(f'[Checkpoint] loaded full checkpoint from {latest_checkpoint} (epoch {start_epoch})')
                logger.info(f'[Checkpoint] loaded from {latest_checkpoint} (resuming from epoch {start_epoch})')
            else:
                model_joint.load_state_dict(checkpoint)
                start_epoch = latest_epoch
                print(f'[Checkpoint] loaded state_dict from {latest_checkpoint} (epoch {start_epoch})')
                print(f'[Checkpoint] Warning: optimizer and scheduler NOT restored (old format)')
                logger.info(f'[Checkpoint] loaded state_dict from {latest_checkpoint} (epoch {start_epoch})')
        except Exception as e:
            print(f'[Checkpoint] Error loading checkpoint: {e}, starting from epoch 0')
            start_epoch = 0
    else:
        print('[Checkpoint] No checkpoint found, starting from epoch 0')
    
    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0
    best_model = None
    
    for epoch_temp in range(start_epoch, epochs):
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model_joint.train()
    
        flag_update = 0
        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()
            scores, diffu_rep, weights, t, item_rep_dis, seq_rep_dis = model_joint(train_batch[0], train_batch[1], train_flag=True)  
            loss_diffu_value = model_joint.loss_diffu_ce(diffu_rep, train_batch[1])
          
            loss_all = loss_diffu_value
            loss_all.backward()
        
            optimizer.step()
            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
                logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
        print("loss in epoch {}: {}".format(epoch_temp, loss_all.item()))
        lr_scheduler.step()

        if (epoch_temp + 1) % 50 == 0:
            save_checkpoint(model_joint, optimizer, epoch_temp + 1, checkpoint_dir, args)

        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model_joint.eval()
            with torch.no_grad():
                metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
                for val_batch in val_data_loader:
                    val_batch = [x.to(device) for x in val_batch]
                    scores_rec, rep_diffu, _, _, _, _ = model_joint(val_batch[0], val_batch[1], train_flag=False)
                    scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu)
                    metrics = hrs_and_ndcgs_k(scores_rec_diffu, val_batch[1], metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)
                        
            for key_temp, values_temp in metrics_dict.items():
                values_mean = round(np.mean(values_temp) * 100, 4)
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update = 1
                    bad_count = 0
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp
                    
            if flag_update == 0:
                bad_count += 1
            else:
                print(best_metrics_dict)
                print(best_epoch)
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
                best_model = copy.deepcopy(model_joint)
            if bad_count >= args.patience:
                print(f'Early stopping at epoch {epoch_temp}')
                logger.info(f'Early stopping at epoch {epoch_temp}')
                break
    
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
        
    if args.eval_interval > epochs:
        best_model = copy.deepcopy(model_joint)
    
    if best_model is not None:
        best_model_path = os.path.join(checkpoint_dir, 'teacher_best.pt')
        torch.save(best_model.state_dict(), best_model_path)
        print(f'[Save] Best model saved to {best_model_path}')
        logger.info(f'[Save] Best model saved to {best_model_path}')
    
    final_model_path = os.path.join(checkpoint_dir, 'teacher_final.pt')
    torch.save(model_joint.state_dict(), final_model_path)
    print(f'[Save] Final model saved to {final_model_path}')
    logger.info(f'[Save] Final model saved to {final_model_path}')
    
    top_100_item = []
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]
            scores_rec, rep_diffu, _, _, _, _ = best_model(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_diffu = best_model.diffu_rep_pre(rep_diffu)
            
            _, indices = torch.topk(scores_rec_diffu, k=100)
            top_100_item.append(indices)

            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1], metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print('Test------------------------------------------------------')
    logger.info('Test------------------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info(test_metrics_dict_mean)
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    print(args)

    if args.diversity_measure:
        path_data = '../datasets/data/category/' + args.dataset +'/id_category_dict.pkl'
        with open(path_data, 'rb') as f:
            id_category_dict = pickle.load(f)
        id_top_100 = torch.cat(top_100_item, dim=0).tolist()
        category_list_100 = []
        for id_top_100_temp in id_top_100:
            category_temp_list = [] 
            for id_temp in id_top_100_temp:
                category_temp_list.append(id_category_dict[id_temp])
            category_list_100.append(category_temp_list)
        category_list_100.append(category_list_100)
        path_data_category = '../datasets/data/category/' + args.dataset +'/DiffuRec_top100_category.pkl'
        with open(path_data_category, 'wb') as f:
            pickle.dump(category_list_100, f)
            
    return best_model, test_metrics_dict_mean