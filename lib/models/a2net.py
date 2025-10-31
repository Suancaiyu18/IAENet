import torch
import torch.nn.functional as F
from torch import nn

INF = 1e8


class Senet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.wl = nn.Conv1d(1, 1, kernel_size=1)
        self.act = nn.Sigmoid()
        nn.init.constant_(self.wl.bias, 0)

    def forward(self, x):
        x = self.pool(x.permute(0, 2, 1))
        x = self.wl(x.permute(0, 2, 1))
        x = self.act(x)
        return x


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


############### Postbackbone ##############
class BaseFeatureNet(nn.Module):
    '''
    Calculate basic feature
    PreBackbobn -> Backbone
    '''
    def __init__(self, cfg):
        super(BaseFeatureNet, self).__init__()
        self.dataset = cfg.DATASET.DATASET_NAME

        self.conv1 = nn.Conv1d(in_channels=cfg.MODEL.IN_FEAT_DIM,
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.conv2 = nn.Conv1d(in_channels=cfg.MODEL.BASE_FEAT_DIM,
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=9, stride=1, padding=4, bias=True)

        self.max_pooling = nn.MaxPool1d(2, 2)
        self.mish = Mish()
        self.down_sampling = cfg.MODEL.B_DOWN_FOUR

    def forward(self, x):
        feat1 = self.conv1(x)
        feat1 = self.mish(feat1)
        feat1 = self.max_pooling(feat1)

        feat2 = self.conv2(feat1)
        feat2 = self.mish(feat2)
        if self.down_sampling:
            feat2 = self.max_pooling(feat2)
        return feat2

############### Neck ##############
class FeatNet(nn.Module):
    '''
    Main network
    Backbone -> Neck
    '''

    def __init__(self, cfg):
        super(FeatNet, self).__init__()
        self.base_feature_net = BaseFeatureNet(cfg)

        # encoder
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=cfg.MODEL.BASE_FEAT_DIM, out_channels=cfg.MODEL.LAYER_DIMS[0],
                                             kernel_size=3, stride=cfg.MODEL.LAYER_STRIDES[0], padding=1), Mish())
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=cfg.MODEL.LAYER_DIMS[0], out_channels=cfg.MODEL.LAYER_DIMS[1],
                                             kernel_size=3, stride=cfg.MODEL.LAYER_STRIDES[1], padding=1), Mish())
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=cfg.MODEL.LAYER_DIMS[1], out_channels=cfg.MODEL.LAYER_DIMS[2],
                                             kernel_size=3, stride=cfg.MODEL.LAYER_STRIDES[2], padding=1), Mish())

        # decoder(top-down)
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=cfg.MODEL.LAYER_DIMS[1], out_channels=cfg.MODEL.LAYER_DIMS[0],
                               stride=2, kernel_size=3, padding=1, output_padding=1, dilation=1))
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=cfg.MODEL.LAYER_DIMS[2], out_channels=cfg.MODEL.LAYER_DIMS[1],
                               stride=2, kernel_size=3, padding=1, output_padding=1, dilation=1))

        # self_processing
        self.up3 = nn.Sequential(
            nn.Conv1d(in_channels=cfg.MODEL.LAYER_DIMS[2], out_channels=cfg.MODEL.LAYER_DIMS[2], kernel_size=3, padding=1),
            nn.Conv1d(in_channels=cfg.MODEL.LAYER_DIMS[2], out_channels=cfg.MODEL.LAYER_DIMS[2] // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cfg.MODEL.LAYER_DIMS[2] // 4, out_channels=cfg.MODEL.LAYER_DIMS[2], kernel_size=1)
        )
        self.up2 = nn.Sequential(
            nn.Conv1d(in_channels=cfg.MODEL.LAYER_DIMS[1], out_channels=cfg.MODEL.LAYER_DIMS[1], kernel_size=3, padding=1),
            nn.Conv1d(in_channels=cfg.MODEL.LAYER_DIMS[1], out_channels=cfg.MODEL.LAYER_DIMS[1] // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cfg.MODEL.LAYER_DIMS[1] // 4, out_channels=cfg.MODEL.LAYER_DIMS[1], kernel_size=1)
        )
        self.up1 = nn.Sequential(
            nn.Conv1d(in_channels=cfg.MODEL.LAYER_DIMS[0], out_channels=cfg.MODEL.LAYER_DIMS[0], kernel_size=3, padding=1),
            nn.Conv1d(in_channels=cfg.MODEL.LAYER_DIMS[0], out_channels=cfg.MODEL.LAYER_DIMS[0] // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cfg.MODEL.LAYER_DIMS[0] // 4, out_channels=cfg.MODEL.LAYER_DIMS[0], kernel_size=1)
        )

        self.senet1 = Senet()
        self.senet2 = Senet()
        self.senet3 = Senet()

        self.up_senet1 = Senet()
        self.up_senet2 = Senet()

        self.transformer = LocTransformer(cfg)

    def forward(self, x):
        results = []
        feat = self.base_feature_net(x)

        feat1 = self.conv1(feat)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)

        feat1 = feat1 * self.senet1(feat1).expand_as(feat1)
        feat2 = feat2 * self.senet2(feat2).expand_as(feat2)
        feat3 = feat3 * self.senet3(feat3).expand_as(feat3)

        up_feat3 = feat3

        up_feat2 = self.upconv2(up_feat3)
        up_feat2 = up_feat2 * self.up_senet2(up_feat2).expand_as(up_feat2)
        up_feat2 = up_feat2 + feat2

        up_feat1 = self.upconv1(up_feat2)
        up_feat1 = up_feat1 * self.up_senet1(up_feat1).expand_as(up_feat1)
        up_feat1 = up_feat1 + feat1

        up_feat3 = self.up3(up_feat3) + up_feat3
        up_feat2 = self.up2(up_feat2) + up_feat2
        up_feat1 = self.up1(up_feat1) + up_feat1

        results.append(up_feat1)
        results.append(up_feat2)
        results.append(up_feat3)

        out = self.transformer(results)

        return out

class ReduceChannel(nn.Module):
    def __init__(self, cfg):
        super(ReduceChannel, self).__init__()
        self.convs = nn.ModuleList()
        for layer in range(cfg.MODEL.NUM_LAYERS):
            conv = nn.Conv1d(cfg.MODEL.LAYER_DIMS[layer], cfg.MODEL.REDU_CHA_DIM, kernel_size=1)
            self.convs.append(conv)
        self.mish = Mish()

    def forward(self, feat_list):
        assert len(feat_list) == len(self.convs)
        results = []
        for i, (conv, feat) in enumerate(zip(self.convs, feat_list)):
            result = conv(feat)
            result = self.mish(result)
            results.append(result)
        return tuple(results)

############### Head ##############
class PredHeadBranch(nn.Module):
    '''
    From ReduceChannel Module
    CAS(ME)^2:
    input: [batch_size, 512, (16,8,4,2)]
    output: Channels reduced into 256
    SAMM:
    input: [batch_size, 512, (128,64,32,16,8,4,2)]
    output: Channels reduced into 256
    '''

    def __init__(self, cfg):
        super(PredHeadBranch, self).__init__()
        self.head_stack_layers = cfg.MODEL.HEAD_LAYERS  # 2
        self._init_head(cfg)

    def _init_head(self, cfg):
        self.convs = nn.ModuleList()
        for layer in range(self.head_stack_layers):
            in_channel = cfg.MODEL.REDU_CHA_DIM
            out_channel = cfg.MODEL.REDU_CHA_DIM
            conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1)
            self.convs.append(conv)
        self.mish = Mish()

    def forward(self, x):
        feat = x
        for conv in self.convs:
            feat = conv(feat)
            feat = self.mish(feat)
        return feat


class LocNet(nn.Module):
    '''
    Predict expression boundary, based on features from different FPN levels
    '''

    def __init__(self, cfg):
        super(LocNet, self).__init__()
        self.reduce_channels = ReduceChannel(cfg)
        self.pred = NewPredHead(cfg)

    def _layer_cal(self, feat_list):
        af_cls = list()
        af_reg = list()

        for feat in feat_list:
            cls_af, reg_af = self.pred(feat)
            af_cls.append(cls_af.permute(0, 2, 1).contiguous())
            af_reg.append(reg_af.permute(0, 2, 1).contiguous())

        af_cls = torch.cat(af_cls, dim=1)  # bs, sum(t_i), n_class+1
        af_reg = torch.cat(af_reg, dim=1)  # bs, sum(t_i), 2
        af_reg = F.relu(af_reg)
        return (af_cls, af_reg)  # 各层预测的结果

    def forward(self, features_list):
        features_list = self.reduce_channels(features_list)
        return self._layer_cal(features_list)


class MLP(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_out = dim_in
        self.drop = nn.Dropout(0.2)
        self.conv1 = nn.Linear(self.dim_out, self.dim_out, bias=False)
        self.ln1 = nn.LayerNorm(self.dim_out)
        self.conv2 = nn.Linear(self.dim_out, self.dim_out, bias=False)
        self.ln2 = nn.LayerNorm(self.dim_out)
        self.mish = nn.GELU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.drop(self.mish(self.drop(self.conv1(self.ln1(self.drop(x))))))
        x = self.ln2(self.drop(self.conv2(x)))
        x = x.permute(0, 2, 1)
        return x


class LocTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlps = nn.ModuleList()
        for i in range(cfg.MODEL.NUM_LAYERS):
            self.mlps.append(MLP(cfg.MODEL.LAYER_DIMS[i]))

    def forward(self, feature_list):
        new_feature_list = []
        for i, trans in enumerate(self.mlps):
            feature = feature_list[i]
            out = trans(feature)
            new_feature_list.append(out)

        return new_feature_list


class NewPredHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.head_branches = nn.ModuleList()
        for _ in range(2):
            self.head_branches.append(PredHeadBranch(cfg))
        num_class = cfg.DATASET.NUM_CLASSES
        af_cls = nn.Conv1d(cfg.MODEL.REDU_CHA_DIM, num_class, kernel_size=3, padding=1)
        af_reg = nn.Conv1d(cfg.MODEL.REDU_CHA_DIM, 2, kernel_size=3, padding=1)

        self.pred_heads = nn.ModuleList([af_cls, af_reg])

    def forward(self, x):
        preds = []
        for pred_branch, pred_head in zip(self.head_branches, self.pred_heads):
            feat = pred_branch(x)
            pred = pred_head(feat)
            preds.append(pred)
        return tuple(preds)


############### All processing ##############
class A2Net(nn.Module):
    def __init__(self, cfg):
        super(A2Net, self).__init__()
        self.features = FeatNet(cfg)
        self.loc_net = LocNet(cfg)

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.)

    def forward(self, x):
        features = self.features(x)
        out_af = self.loc_net(features)

        return out_af


if __name__ == '__main__':
    import sys

    sys.path.append("/home/ljj/lxd/LGSNet/LGSNet/lib/")
    from config import cfg, update_config

    cfg_file = "/home/ljj/lxd/LGSNet/LGSNet/experiments/cas.yaml"
    update_config(cfg_file)

    model = A2Net(cfg).cuda()
    data = torch.randn((64, 2048, 64)).cuda()
    mask = torch.ones((64, 1, 64), dtype=torch.bool).cuda()
    output = model(data)
