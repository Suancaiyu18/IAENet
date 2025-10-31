import os
import numpy as np
from torch.utils.data import Dataset


class TALDataset(Dataset):
    def __init__(self, cfg, split, subject):
        self.root = os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.FEAT_DIR, subject)
        self.split = split
        self.train_split = cfg.DATASET.TRAIN_SPLIT
        self.max_segment_num = cfg.DATASET.MAX_SEGMENT_NUM
        self.base_dir = os.path.join(self.root, self.split)
        self.datas = self._make_dataset()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        file_name = self.datas[idx]
        data = np.load(os.path.join(self.base_dir, file_name))

        feat_tem = data['feat_tem']
        feat_spa = data['feat_spa']
        begin_frame = data['begin_frame']
        # pass video_name vis list
        video_name = str(data['vid_name'])

        if self.split == self.train_split:
            action = data['action']
            action = np.array(action).astype('float32')
            label = data['class_label']
            num_segment = action.shape[0]
            assert num_segment > 0, 'no action in {}!!!'.format(video_name)
            action_padding = np.zeros((self.max_segment_num, 2), dtype=np.float)
            action_padding[:num_segment, :] = action
            label_padding = np.zeros(self.max_segment_num, dtype=np.int)
            label_padding[:num_segment] = label

            return feat_spa, feat_tem, action_padding, label_padding, num_segment
        else:
            return feat_spa, feat_tem, begin_frame, video_name

    def _make_dataset(self):
        datas = os.listdir(self.base_dir)
        datas = [i for i in datas if i.endswith('.npz')]
        return datas

