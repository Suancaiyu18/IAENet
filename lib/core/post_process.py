import os
from core.nms import temporal_nms_all, wbf_nms

def final_result_process(out_df, epoch, subject, cfg):
    path_tmp = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject, cfg.TEST.PREDICT_TXT_FILE)
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    res_txt_file = os.path.join(path_tmp, 'test_' + str(epoch).zfill(2) + '.txt')
    if os.path.exists(res_txt_file):
        os.remove(res_txt_file)

    f = open(res_txt_file, 'a')
    df_af = out_df
    df_name = df_af

    video_name_list = list(set(df_name.video_name.values[:]))

    for video_name in video_name_list:
        tmpdf = df_af[df_af.video_name == video_name]

        if epoch >= 15:
            if cfg.TEST.NMS_FALG == 0:
                df_nms = temporal_nms_all(tmpdf, cfg)
            elif cfg.TEST.NMS_FALG == 1:
                df_nms = wbf_nms(tmpdf, cfg)
        else:
            df_nms = temporal_nms_all(tmpdf, cfg)
        # ensure there are most 200 proposals
        df_vid = df_nms.sort_values(by='score', ascending=False)
        for i in range(len(df_vid)):
            start_time = df_vid.start.values[i]
            end_time = df_vid.end.values[i]
            cate_idx = df_vid.cate_idx.values[i]
            try:
                label = df_vid.label.values[i]
                strout = '%s\t%.3f\t%.3f\t%d\t%.4f\t%.1f\n' % (
                video_name, float(start_time), float(end_time), label, df_vid.score.values[i], cate_idx)
                f.write(strout)
            except:
                strout = '%s\t%.3f\t%.3f\t%.4f\t%.1f\n' % (
                video_name, float(start_time), float(end_time), df_vid.score.values[i], cate_idx)
                f.write(strout)

    f.close()
