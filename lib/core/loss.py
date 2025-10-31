import torch
import torch.nn as nn

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()

def one_hot_embedding(labels, num_classes):
    y_tmp = torch.eye(num_classes, device=labels.device)
    return y_tmp[labels]

class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=20, eps=1e-6):
        super(Focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, x, y):
        t = one_hot_embedding(y, 1 + self.num_classes)
        t = t[:, 1:]

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        pt = pt.clamp(min=self.eps)  # avoid log(0)
        self.alpha = torch.tensor(self.alpha).cuda()
        w = self.alpha * t + (1 - self.alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        loss = -(w * (1 - pt).pow(self.gamma) * torch.log(pt))
        return loss.sum()

def iou_loss(pred, target, enhance=None):
    inter_min = torch.max(pred[:, 0], target[:, 0])
    inter_max = torch.min(pred[:, 1], target[:, 1])
    inter_len = (inter_max - inter_min).clamp(min=1e-6)
    union_len = (pred[:, 1] - pred[:, 0]) + (target[:, 1] - target[:, 0]) - inter_len
    tious = inter_len / union_len
    if enhance is not None:
        loss = (enhance * (1 - tious)).mean()
    else:
        loss = (1 - tious).mean()
    return loss

def loss_function_af(cate_label, preds_cls, target_loc, pred_loc, allstri, reg_real, cfg):
    '''
        preds_cls: bs, t1+t2+..., n_class
        pred_regs_batch: bs, t1+t2+..., 2
        '''
    batch_size = preds_cls.size(0)
    cate_label_view = cate_label.view(-1)
    cate_label_view = cate_label_view.type_as(dtypel)
    preds_cls_view = preds_cls.view(-1, cfg.DATASET.NUM_CLASSES)
    pmask = (cate_label_view > 0).type_as(dtype)

    if torch.sum(pmask) > 0:
        # regression loss
        mask = pmask == 1.0
        proposals = cate_label_view[mask]
        micro = proposals == 2
        macro = proposals == 1
        enhance = cfg.TRAIN.ENHANCE_TIMES * torch.ones_like(proposals) * micro + torch.ones_like(proposals) * macro
        pred_loc_reg = pred_loc.view(-1, 2)[mask]
        target_loc_reg = target_loc.view(-1, 2)[mask]
        reg_loss = iou_loss(pred_loc_reg, target_loc_reg, enhance)
    else:
        reg_loss = torch.tensor(0.).type_as(dtype)
    # cls loss
    cate_loss_f = Focal_loss(alpha=cfg.TRAIN.ALPHA, num_classes=cfg.DATASET.NUM_CLASSES)
    cate_loss = cate_loss_f(preds_cls_view, cate_label_view) / (torch.sum(pmask) + batch_size)

    reg_loss = reg_loss
    return cate_loss, reg_loss
