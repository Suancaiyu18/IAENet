import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import time
import math
import argparse
import random
import _init_paths
from config import cfg
from config import update_config
import pprint
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset.TALDataset import TALDataset
from models.a2net import A2Net
from core.function import train, evaluation
from core.post_process import final_result_process
from utils.utils import save_model, backup_codes

# fix random seed
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

# for reproduction, same as orig. paper setting
seed_torch()

def main(subject, config):
    update_config(config)
    # create output directory
    if cfg.BASIC.CREATE_OUTPUT_DIR:
        out_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    # copy config file
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject, 'code')
        backup_codes(cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LISTS)

    # model
    model = A2Net(cfg)
    model.cuda()

    # optimizer
    # warm_up_with_cosine_lr
    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    warm_up_with_cosine_lr = lambda \
        epoch: epoch / cfg.TRAIN.WARM_UP_EPOCH if epoch <= cfg.TRAIN.WARM_UP_EPOCH else 0.5 * (math.cos(
        (epoch - cfg.TRAIN.WARM_UP_EPOCH) / (cfg.TRAIN.END_EPOCH - cfg.TRAIN.WARM_UP_EPOCH) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    # data loader
    train_dset = TALDataset(cfg, cfg.DATASET.TRAIN_SPLIT, subject)
    train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True, drop_last=False, num_workers=cfg.BASIC.WORKERS,
                              pin_memory=cfg.DATASET.PIN_MEMORY)
    val_dset = TALDataset(cfg, cfg.DATASET.VAL_SPLIT, subject)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE,
                            shuffle=False, drop_last=False, num_workers=cfg.BASIC.WORKERS,
                            pin_memory=cfg.DATASET.PIN_MEMORY)

    # creating log path
    log_path = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject,
                            str(cfg.TRAIN.BATCH_SIZE) + '_' + cfg.TRAIN.LOG_FILE)
    if os.path.exists(log_path):
        os.remove(log_path)

    for epoch in range(cfg.TRAIN.END_EPOCH):
        loss_train, cls_loss_af, reg_loss_af = train(cfg, train_loader, model, optimizer)
        print('Epoch %d: loss: %.4f AF cls loss: %.4f, reg loss: %.4f ' % (
            epoch, loss_train, cls_loss_af, reg_loss_af))

        with open(log_path, 'a') as f:
            f.write('Epoch %d: loss: %.4f AF cls loss: %.4f, reg loss: %.4f\n' % (
                epoch, loss_train, cls_loss_af, reg_loss_af))

        # decay lr
        scheduler.step()

        if (epoch + 1) % cfg.TEST.EVAL_INTERVAL == 0:
            out_df_af = evaluation(val_loader, model, cfg)
            out_df_list = out_df_af
            final_result_process(out_df_list, epoch, subject, cfg,)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MER SPOT')
    parser.add_argument('--cfg', type=str, help='experiment config file',
                        default="/home/geek/spot/New_Code_2_3/IAENet_CNN/experiments/cas2.yaml")
    parser.add_argument('--dataset', type=str, default='cas(me)^2')
    args = parser.parse_args()

    dataset = args.dataset
    new_cfg = args.cfg
    print(args.cfg)
    if dataset == 'cas(me)^2':
        ca_subject = [15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40]
        ca_subject = ['s' + str(i) for i in ca_subject]
        for i in ca_subject:
            print(i)
            subject = 'subject_' + i
            main(subject, new_cfg)
    else:
        sa_subject = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,28,30,32,33,34,35,36,37,99]
        sa_subject = [str(i).zfill(3) for i in sa_subject]
        for i in sa_subject:
            print(i)
            subject = 'subject_' + i
            main(subject, new_cfg)