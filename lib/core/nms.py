import numpy as np
import pandas as pd

def tiou(anchors_min, anchors_max, len_anchors, box_min, box_max):
    '''
    calculate jaccatd score between a box and an anchor
    '''
    inter_xmin = np.maximum(anchors_min, box_min)
    inter_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(inter_xmax-inter_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    tiou = np.divide(inter_len, union_len)
    return tiou

def temporal_nms_all(df, cfg):
    # returned values
    rstart = list()
    rend = list()
    rscore = list()
    rcate_idx = list()

    mae_indices = list() # 1
    mae_cate_idx = list() # 1
    me_indices = list() # 2
    me_cate_idx = list() # 2

    # attention: for THUMOS, a sliding window may contain actions from different class
    tmp_df = df
    cate_idx = np.array(tmp_df.cate_idx.values[:])
    for i, j in enumerate(cate_idx):
        if j == 1:
            mae_indices.append(i)
            mae_cate_idx.append(j)
        else:
            me_indices.append(i)
            me_cate_idx.append(j)

    mae_start_time = np.array(tmp_df.xmin.values[:])[mae_indices]
    mae_end_time = np.array(tmp_df.xmax.values[:])[mae_indices]
    mae_scores = np.array(tmp_df.conf.values[:])[mae_indices]
    mae_duration = mae_end_time - mae_start_time
    mae_order = mae_scores.argsort()[::-1]

    me_start_time = np.array(tmp_df.xmin.values[:])[me_indices]
    me_end_time = np.array(tmp_df.xmax.values[:])[me_indices]
    me_scores = np.array(tmp_df.conf.values[:])[me_indices]
    me_duration = me_end_time - me_start_time
    me_order = me_scores.argsort()[::-1]

    mae_keep = list()
    while (mae_order.size > 0) and (len(mae_keep) < cfg.TEST.TOP_K_RPOPOSAL_MAE):
        i = mae_order[0]
        mae_keep.append(i)
        tt1 = np.maximum(mae_start_time[i], mae_start_time[mae_order[1:]])
        tt2 = np.minimum(mae_end_time[i], mae_end_time[mae_order[1:]])
        intersection = tt2 - tt1
        union = (mae_duration[i] + mae_duration[mae_order[1:]] - intersection).astype(float)
        iou = intersection / union

        inds = np.where(iou <= cfg.TEST.NMS_TH)[0]
        mae_order = mae_order[inds + 1]
    # record the result
    for idx in mae_keep:
        rstart.append(float(mae_start_time[idx]))
        rend.append(float(mae_end_time[idx]))
        rscore.append(mae_scores[idx])
        rcate_idx.append(mae_cate_idx[idx])

    me_keep = list()
    while (me_order.size > 0) and (len(me_keep) < cfg.TEST.TOP_K_RPOPOSAL_ME):
        i = me_order[0]
        me_keep.append(i)
        tt1 = np.maximum(me_start_time[i], me_start_time[me_order[1:]])
        tt2 = np.minimum(me_end_time[i], me_end_time[me_order[1:]])
        intersection = tt2 - tt1
        union = (me_duration[i] + me_duration[me_order[1:]] - intersection).astype(float)
        iou = intersection / union

        inds = np.where(iou <= cfg.TEST.NMS_TH)[0]
        me_order = me_order[inds + 1]
    # record the result
    for idx in me_keep:
        rstart.append(float(me_start_time[idx]))
        rend.append(float(me_end_time[idx]))
        rscore.append(me_scores[idx])
        rcate_idx.append(me_cate_idx[idx])

    new_df = pd.DataFrame()
    # 如果判断区间<=3,则将其舍弃
    new_rstart = []
    new_rend = []
    new_rscore = []
    new_rcate_idx = []
    for i, (rs, re, rsc, rci) in enumerate(zip(rstart, rend, rscore, rcate_idx)):
        if re-rs >= 4:
            new_rstart.append(rs)
            new_rend.append(re)
            new_rscore.append(rsc)
            new_rcate_idx.append(rci)

    new_df['start'] = new_rstart
    new_df['end'] = new_rend
    new_df['score'] = new_rscore
    new_df['cate_idx'] = new_rcate_idx

    return new_df


def wbf_nms(df, cfg):
    # adjust conf for each pred
    iou_thr = cfg.TEST.WBF_NMS_TH
    skip_box_thr = cfg.TEST.WBF_SKIP_TH
    df = df.sort_values(by='conf', ascending=False)
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    # 3 dimensions: models_number, model_preds, 2
    # expect to normlize boxes to [0,1]
    boxes_list = [[list(i) for i in list(zip(tstart, tend))]]
    conf_list = [list(df.conf.values[:])]
    labels_list = [list(df.cate_idx.values[:])]

    boxes, scores, cate_idx = weighted_boxes_fusion(cfg, boxes_list, conf_list, labels_list,
                                                    weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    new_df = pd.DataFrame()
    # 如果判断区间<=3,则将其舍弃
    new_rstart = []
    new_rend = []
    new_rscore = []
    new_cate_idx = []
    for i, (rs, re, rsc, rci) in enumerate(zip(boxes[:,0], boxes[:,1], scores, cate_idx)):
        if re - rs >= 4:
            new_rstart.append(rs)
            new_rend.append(re)
            new_rscore.append(rsc)
            new_cate_idx.append(rci)

    new_df['start'] = new_rstart
    new_df['end'] = new_rend
    new_df['score'] = new_rscore
    new_df['cate_idx'] = new_cate_idx

    return new_df


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()
    thr1 = thr[0] # mae
    thr2 = thr[1] # me
    # each model
    for t in range(len(boxes)):
        # all pred_loc in the each model
        for j in range(len(boxes[t])):
            score = scores[t][j]
            label = int(labels[t][j])
            if label == 1.0:
                if score < thr1:
                    continue
            if label == 2.0:
                if score < thr2:
                    continue
            box_part = boxes[t][j]
            b = [int(label), float(score) * weights[t], float(box_part[0]), float(box_part[1])]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]
    return new_boxes


def get_weighted_box(type, cfg, boxes, conf_type='avg'):
    box = np.zeros(4, dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        # bconf_tmp = b[1]
        # box[2:] += (b[1] * b[2:])
        if type == 1.0: # mae
            bconf_tmp = np.power(b[1], cfg.TEST.WBF_REWEIGHT_FACTOR_MAE)
            box[2:] += (bconf_tmp * b[2:])
            conf += bconf_tmp
            conf_list.append(bconf_tmp)
        else: # me
            bconf_tmp = np.power(b[1], cfg.TEST.WBF_REWEIGHT_FACTOR_ME)
            box[2:] += (bconf_tmp* b[2:])
            conf += bconf_tmp
            conf_list.append(bconf_tmp)
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2:] /= conf
    return box


def iou_temporal(box, nbox):
    inter_xmin = np.maximum(box[0], nbox[0])
    inter_xmax = np.minimum(box[1], nbox[1])
    inter_len = np.maximum(inter_xmax-inter_xmin, 0.)
    union_len = (box[1]-box[0]) + (nbox[1]-nbox[0]) - inter_len
    tiou = np.divide(inter_len, union_len)
    return tiou


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = iou_temporal(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion(cfg, boxes_list, scores_list, labels_list, weights=None,
                          iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(label, cfg, new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            if not allows_overflow:
                weighted_boxes[i][1] = weighted_boxes[i][1] * min(weights.sum(), len(new_boxes[i])) / weights.sum()
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(new_boxes[i]) / weights.sum()
        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels