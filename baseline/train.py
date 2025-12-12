# -*- coding: utf-8 -*-
import os, sys, random, logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from Dataset import BaseDataSets, RandomGenerator, ValGenerator, TwoStreamBatchSampler
from unet import UNet
from utils import losses, ramps
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score


class Args:
    root_path = "/mnt/data1/mxl/home/SSL4MIS-master_h5/data"
    exp = "Carotid_MeanTeacher_FIXED_20_24_8"
    model = "unet"
    max_iterations = 30000
    batch_size = 24
    base_lr = 0.01
    patch_size = [256, 256]
    seed = 1337
    num_classes = 3      
    num_cls = 2          
    labeled_bs = 8    
    ema_decay = 0.99
    seg_consistency = 0.1     
    cls_consistency = 0.1  
    consistency_rampup = 200.0
    deterministic = 1
    num_labeled = 200

args = Args()

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

def get_current_consistency_weight(epoch):

    return args.seg_consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):

    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train(args, snapshot_path):
    base_lr = args.base_lr
    batch_size = args.batch_size

    def create_model(ema=False):
        model = UNet(in_chns=1, seg_classes=args.num_classes, cls_classes=args.num_cls)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model().to(device)
    ema_model = create_model(ema=True).to(device)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        
    db_train = BaseDataSets(base_dir=args.root_path, split="train",transform=RandomGenerator(args.patch_size),num_labeled=args.num_labeled)
                             
    db_val = BaseDataSets(base_dir=args.root_path, split="val",transform=ValGenerator(args.patch_size))
    
    total = len(db_train)
    labeled = db_train.num_labeled
    print(f"[INFO] Total slices: {total}, Labeled slices: {labeled}")
    
    labeled_idxs = list(range(0, labeled))
    unlabeled_idxs = list(range(labeled, total))
    
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                                          batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train()
    
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    dice_loss = losses.DiceLoss(args.num_classes)
    ce_loss = nn.CrossEntropyLoss()
    cls_loss_fn = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info(f"{len(trainloader)} iterations per epoch")

    iter_num = 0
    max_epoch = args.max_iterations // len(trainloader) + 1
    best_plaque_dice, best_score = 0, 0

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        
        for i_batch, batch in enumerate(trainloader):
            img_long = batch["image_long"].to(device)
            img_trans = batch["image_trans"].to(device)
            label_long = batch["label_long"].to(device)
            label_trans = batch["label_trans"].to(device)
            cls_label = batch["cls_label"].long().to(device)
            
            label_l_long = label_long[:args.labeled_bs]
            label_l_trans = label_trans[:args.labeled_bs]
            cls_l_label = cls_label[:args.labeled_bs]
            
            img_u_long = img_long[args.labeled_bs:]
            img_u_trans = img_trans[args.labeled_bs:]

            noise_long = torch.clamp(torch.randn_like(img_u_long) * 0.1, -0.2, 0.2)
            noise_trans = torch.clamp(torch.randn_like(img_u_trans) * 0.1, -0.2, 0.2)
            ema_long_input = img_u_long + noise_long
            ema_trans_input = img_u_trans + noise_trans


            # Student
            seg_long_output, seg_trans_output, cls_output = model(img_long, img_trans)
            seglong_soft = torch.softmax(seg_long_output, dim=1)
            segtrans_soft = torch.softmax(seg_trans_output, dim=1)
            
            # Teacher
            with torch.no_grad():
                ema_seg_long, ema_seg_trans, ema_cls = ema_model(ema_long_input, ema_trans_input)
                ema_seglong_soft = torch.softmax(ema_seg_long, dim=1)
                ema_segtrans_soft = torch.softmax(ema_seg_trans, dim=1)

            ce_loss_long = ce_loss(seg_long_output[:args.labeled_bs], label_l_long.long())
            ce_loss_trans = ce_loss(seg_trans_output[:args.labeled_bs], label_l_trans.long())
            dice_long = dice_loss(seglong_soft[:args.labeled_bs], label_l_long.unsqueeze(1))
            dice_trans = dice_loss(segtrans_soft[:args.labeled_bs], label_l_trans.unsqueeze(1))
            
            sup_loss_seg = 0.5 * (ce_loss_long + ce_loss_trans + dice_long + dice_trans)
            sup_loss_cls = cls_loss_fn(cls_output[:args.labeled_bs], cls_l_label)
            
            supervised_loss = sup_loss_seg + sup_loss_cls

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            
            if iter_num < 1000:
                consistency_loss = 0.0
            else:
                consistency_loss_seg = 0.5 * (
                    torch.mean((seglong_soft[args.labeled_bs:] - ema_seglong_soft) ** 2) +
                    torch.mean((segtrans_soft[args.labeled_bs:] - ema_segtrans_soft) ** 2)
                )
                
                cls_soft = torch.softmax(cls_output[args.labeled_bs:], dim=1)
                ema_cls_soft = torch.softmax(ema_cls, dim=1)
                consistency_loss_cls = torch.mean((cls_soft - ema_cls_soft) ** 2)
                
                consistency_loss = consistency_loss_seg + consistency_loss_cls

            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            
            lr_ = base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            
            # ====================
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/total_loss', loss, iter_num)
            writer.add_scalar('train/sup_loss', supervised_loss, iter_num)
            writer.add_scalar('train/sup_loss_seg', sup_loss_seg, iter_num)
            writer.add_scalar('train/sup_loss_cls', sup_loss_cls, iter_num)
            writer.add_scalar('train/consistency_loss', 
                consistency_loss if isinstance(consistency_loss, float) else consistency_loss.item(), 
                iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)

            if iter_num % 250 == 0:
                logging.info(
                    f'Iter {iter_num}: loss={loss.item():.4f}, '
                    f'sup_seg={sup_loss_seg.item():.4f}, '
                    f'sup_cls={sup_loss_cls.item():.4f}, '
                    f'consistency={consistency_loss if isinstance(consistency_loss, float) else consistency_loss.item():.4f}, '
                    f'cons_weight={consistency_weight:.4f}')
                
                writer.add_image('train/image_long', img_long[0], iter_num)
                writer.add_image('train/label_long', label_long[0:1].float() * 50, iter_num)
                pred_long = torch.argmax(seg_long_output[0:1], dim=1, keepdim=True).float()
                pred_trans = torch.argmax(seg_trans_output[0:1], dim=1, keepdim=True).float()
                writer.add_image('train/pred_long', pred_long[0] * 50, iter_num)
                writer.add_image('train/pred_trans', pred_trans[0] * 50, iter_num)
                
            # ==========val==========
            if iter_num % 500 == 0:
                model.eval()
                
                dice_vessel_list_long, dice_plaque_list_long = [], []
                dice_vessel_list_trans, dice_plaque_list_trans = [], []
                all_preds, all_gts, all_probs = [], [], []

                for v_idx, vbatch in enumerate(valloader):
                    v_img_long = vbatch["image_long"].to(device)
                    v_img_trans = vbatch["image_trans"].to(device)
                    v_label_long = vbatch["label_long"].to(device)
                    v_label_trans = vbatch["label_trans"].to(device)
                    v_cls = vbatch["cls_label"].to(device)
                    
                    with torch.no_grad():
                        seg_l, seg_t, cls_p = model(v_img_long, v_img_trans)
                        seg_pred = torch.argmax(seg_l, dim=1)
                        seg_pred_trans = torch.argmax(seg_t, dim=1)
                        
                        for c in range(1, args.num_classes):
                            pred_c = (seg_pred == c).float()
                            gt_c = (v_label_long == c).float()
                            intersection = (pred_c * gt_c).sum()
                            dice = (2 * intersection) / (pred_c.sum() + gt_c.sum() + 1e-5)
                            
                            if c == 1:
                                dice_vessel_list_long.append(dice.item())
                            elif c == 2:
                                dice_plaque_list_long.append(dice.item())
                        
                        for c in range(1, args.num_classes):
                            pred_c = (seg_pred_trans == c).float()
                            gt_c = (v_label_trans == c).float()
                            intersection = (pred_c * gt_c).sum()
                            dice = (2 * intersection) / (pred_c.sum() + gt_c.sum() + 1e-5)
                            
                            if c == 1:
                                dice_vessel_list_trans.append(dice.item())
                            elif c == 2:
                                dice_plaque_list_trans.append(dice.item())
                        
                        cls_pred = torch.argmax(cls_p, dim=1)
                        cls_prob = torch.softmax(cls_p, dim=1)[:, 1]
                        all_preds.extend(cls_pred.cpu().numpy())
                        all_gts.extend(v_cls.cpu().numpy())
                        all_probs.extend(cls_prob.cpu().numpy())

                mean_dice_vessel_long = np.mean(dice_vessel_list_long)
                mean_dice_plaque_long = np.mean(dice_plaque_list_long)
                mean_dice_long = (mean_dice_vessel_long + mean_dice_plaque_long) / 2
                
                mean_dice_vessel_trans = np.mean(dice_vessel_list_trans)
                mean_dice_plaque_trans = np.mean(dice_plaque_list_trans)
                mean_dice_trans = (mean_dice_vessel_trans + mean_dice_plaque_trans) / 2
                
                mean_dice_vessel = (mean_dice_vessel_long + mean_dice_vessel_trans) / 2
                mean_dice_plaque = (mean_dice_plaque_long + mean_dice_plaque_trans) / 2
                mean_dice = (mean_dice_long + mean_dice_trans) / 2
                
                all_preds = np.array(all_preds)
                all_gts = np.array(all_gts)
                all_probs = np.array(all_probs)

                f1 = f1_score(all_gts, all_preds, zero_division=0)
                try:
                    roc_auc = roc_auc_score(all_gts, all_probs)
                except ValueError:
                    roc_auc = 0.0
                kappa = cohen_kappa_score(all_gts, all_preds)
                score = (f1 + roc_auc + kappa) / 3.0
                
                writer.add_scalar('val/dice_vessel_long', mean_dice_vessel_long, iter_num)
                writer.add_scalar('val/dice_plaque_long', mean_dice_plaque_long, iter_num)
                writer.add_scalar('val/dice_long', mean_dice_long, iter_num)
                writer.add_scalar('val/dice_vessel_trans', mean_dice_vessel_trans, iter_num)
                writer.add_scalar('val/dice_plaque_trans', mean_dice_plaque_trans, iter_num)
                writer.add_scalar('val/dice_trans', mean_dice_trans, iter_num)
                writer.add_scalar('val/dice_vessel', mean_dice_vessel, iter_num)
                writer.add_scalar('val/dice_plaque', mean_dice_plaque, iter_num)
                writer.add_scalar('val/mean_dice', mean_dice, iter_num)
                writer.add_scalar('val/cls_score', score, iter_num)
                writer.add_scalar('val/f1', f1, iter_num)
                writer.add_scalar('val/auc', roc_auc, iter_num)
                writer.add_scalar('val/kappa', kappa, iter_num)
                
                logging.info(
                    f'Val Iter {iter_num}:\n'
                    f'  Long-axis:  vessel={mean_dice_vessel_long:.4f}, plaque={mean_dice_plaque_long:.4f}, mean={mean_dice_long:.4f}\n'
                    f'  Trans-axis: vessel={mean_dice_vessel_trans:.4f}, plaque={mean_dice_plaque_trans:.4f}, mean={mean_dice_trans:.4f}\n'
                    f'  Combined:   vessel={mean_dice_vessel:.4f}, plaque={mean_dice_plaque:.4f}, mean={mean_dice:.4f}\n'
                    f'  Classification: F1={f1:.4f}, AUC={roc_auc:.4f}, Kappa={kappa:.4f}, Score={score:.4f}')
                
                if mean_dice_plaque > best_plaque_dice:
                    best_plaque_dice = mean_dice_plaque
                    save_path = os.path.join(snapshot_path, f"best_dice_{iter_num}.pth")
                    torch.save(model.state_dict(), save_path)
                    logging.info(f" Saved best dice model: {save_path}")
                
                if score > best_score:
                    best_score = score
                    save_path = os.path.join(snapshot_path, f"best_score_{iter_num}.pth")
                    torch.save(model.state_dict(), save_path)
                    logging.info(f" Saved best score model: {save_path}")
                
                model.train()

            if iter_num % 3000 == 0:
                save_path = os.path.join(snapshot_path, f'iter_{iter_num}.pth')
                torch.save(model.state_dict(), save_path)
                logging.info(f" Saved checkpoint: {save_path}")

            if iter_num >= args.max_iterations:
                break
        
        if iter_num >= args.max_iterations:
            break

    writer.close()
    logging.info(f" Training finished! Best Dice: {best_plaque_dice:.4f}, Best Score: {best_score:.4f}")
    return "Training Finished!"

if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    snapshot_path = f"/mnt/data1/mxl/home/SSL4MIS-master_h5/result/{args.exp}"
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("=" * 50)
    logging.info("Training Configuration:")
    logging.info("=" * 50)
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")
    logging.info("=" * 50)

    train(args, snapshot_path)