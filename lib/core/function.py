import numpy as np
import torch
import pandas as pd

from core.loss import loss_function_af
from core.utils_box import reg2loc
from core.utils_sf import get_targets_af, get_points


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def train(cfg, train_loader, model, optimizer):
    model.train()
    cls_loss_af_record, reg_loss_af_record, loss_record = 0, 0, 0
    for feat_spa, feat_tem, boxes, label, action_num in train_loader:  #action_num指的是这段窗口中含有几个表情
        optimizer.zero_grad()
        # optical flow + rgb
        feat_spa = feat_spa.type_as(dtype)
        feat_tem = feat_tem.type_as(dtype)
        feature = torch.cat((feat_spa, feat_tem), dim=1)
        boxes = boxes.float().type_as(dtype)
        label = label.type_as(dtypel)

        out_af = model(feature)

        cate_label, reg_label, allstri = get_targets_af(cfg, boxes, label, action_num)
        reg_label = reg_label.type_as(dtype)  # bs, sum(t_i), 2
        cate_label = cate_label.type_as(dtype)

        preds_cls, preds_reg = out_af

        preds_loc = reg2loc(cfg, preds_reg)
        target_loc = reg2loc(cfg, reg_label)

        cls_loss_af, reg_loss_af = loss_function_af(cate_label, preds_cls, target_loc, preds_loc,
                                                    allstri, reg_label, cfg)
        if cfg.TRAIN.ENHANCE_REG > 0:
            loss = cls_loss_af + cfg.TRAIN.ENHANCE_REG * reg_loss_af
        else:
            loss_weight = cls_loss_af.detach() / max(reg_loss_af.item(), 0.01)
            loss = cls_loss_af + loss_weight * reg_loss_af
        loss.backward()
        optimizer.step()

        cls_loss_af_record += cls_loss_af.item()
        reg_loss_af_record += reg_loss_af.item()
        loss_record = loss_record + loss.item()

    return loss_record / len(train_loader), cls_loss_af_record / len(train_loader), \
           reg_loss_af_record / len(train_loader)


def evaluation(val_loader, model, cfg):
    model.eval()
    strides = [
        torch.tensor(cfg.MODEL.TEMPORAL_STRIDE[i]).expand(
            cfg.MODEL.TEMPORAL_LENGTH[i]) for i in range(cfg.MODEL.NUM_LAYERS)  # n_point
    ]
    strides = torch.cat(strides).type_as(dtype)  # sum_i(t_i),

    out_df_af = pd.DataFrame(columns=cfg.TEST.OUTDF_COLUMNS_AF)
    for feat_spa, feat_tem, begin_frame, video_name in val_loader:
        begin_frame = begin_frame.detach().numpy()
        # optical flow + rgb
        feat_spa = feat_spa.type_as(dtype)
        feat_tem = feat_tem.type_as(dtype)
        feature = torch.cat((feat_spa, feat_tem), dim=1)    # 这一步是将光流和帧二者信息进行简单的相加
        out_af = model(feature)
        # ##########################anchor_free############################################################
        preds_cls, preds_reg = out_af
        preds_cls = preds_cls.sigmoid()
        if cfg.MODEL.NORM_ON_BBOX:
            assert strides.size(0) == preds_reg.size(1)
            preds_reg = preds_reg * strides[None, :, None].expand_as(preds_reg)
        preds_loc = reg2loc(cfg, preds_reg)

        preds_cls = preds_cls.detach().cpu().numpy()
        xmins = preds_loc[..., 0]
        xmins = xmins.detach().cpu().numpy()
        xmaxs = preds_loc[..., 1]
        xmaxs = xmaxs.detach().cpu().numpy()

        tmp_df_af = result_process_af(video_name, begin_frame, preds_cls, xmins, xmaxs, cfg)
        out_df_af = pd.concat([out_df_af, tmp_df_af], sort=True)

    return out_df_af


def result_process_af(video_names, start_frames, cls_scores, anchors_xmin, anchors_xmax, cfg):
    # anchors_class,... : bs, sum_i(t_i*n_box), n_class
    # anchors_xmin, anchors_xmax: bs, sum_i(t_i*n_box)
    # video_names, start_frames: bs,
    out_df = pd.DataFrame()

    frame_window_width = cfg.DATASET.WINDOW_SIZE
    xmins = np.maximum(anchors_xmin, 0)
    xmins = xmins + np.expand_dims(start_frames, axis=1)
    xmaxs = np.minimum(anchors_xmax, frame_window_width)
    xmaxs = xmaxs + np.expand_dims(start_frames, axis=1)

    # expand video_name
    vid_name_df = list()
    num_tem_loc = anchors_xmin.shape[1]
    for i in range(len(video_names)):
        vid_names = [video_names[i]] * num_tem_loc
        vid_name_df.extend(vid_names)
    out_df['video_name'] = vid_name_df

    # reshape numpy array
    # Notice: this is not flexible enough
    num_element = anchors_xmin.shape[0] * anchors_xmin.shape[1]
    xmins_tmp = np.reshape(xmins, num_element)
    out_df['xmin'] = xmins_tmp
    xmaxs_tmp = np.reshape(xmaxs, num_element)
    out_df['xmax'] = xmaxs_tmp

    scores_action = cls_scores
    max_values = np.amax(scores_action, axis=2)
    conf_tmp = np.reshape(max_values, num_element)
    out_df['conf'] = conf_tmp
    max_idxs = np.argmax(scores_action, axis=2)
    max_idxs = max_idxs + 1
    cate_idx_tmp = np.reshape(max_idxs, num_element)
    # Notice: convert index into category type
    class_real = cfg.DATASET.CLASS_IDX
    for i in range(len(cate_idx_tmp)):
        cate_idx_tmp[i] = class_real[int(cate_idx_tmp[i])]
    out_df['cate_idx'] = cate_idx_tmp

    out_df = out_df[cfg.TEST.OUTDF_COLUMNS_AF]
    return out_df