import os
import glob
import pandas as pd
import numpy as np
import argparse


def all_score(TP1, TP2, N1, N2, recall_all):
    if (N1 == 0 and N2 == 0) or (TP1 == 0 and TP2 == 0):
        precision_all = 0
        F1_SCORE = 0
    else:
        precision_all = 1.0 * (TP1 + TP2) / (N1 + N2)
        F1_SCORE = 2 * (recall_all * precision_all) / (recall_all + precision_all)
    return F1_SCORE, precision_all

def proposal_arrange(test_path, k, ann_csv, iou_threshold, mae_or_me):
    N, TP = 0, 0
    all_tp_sample_list = []
    for ip in test_path:
        with open(ip, 'r') as f:
            all_lines = f.readlines()
        if not all_lines:
            continue
        all_lines = [h.split('\t') for h in all_lines]
        # divide all gts of every video
        count = 1
        tmp_list = list()
        all_test = dict()
        all_video = list(set([name[0] for name in all_lines]))
        for tv in all_video:
            tmp_video = tv
            for j in range(len(all_lines)):
                if all_lines[j][0] == tmp_video:
                    tmp_list.append(all_lines[j])
            all_test[count] = tmp_list
            count = count + 1
            tmp_list = list()
        # number of GT of every video
        num_of_video = len(all_test.keys())
        # least len of GT
        part_tmp = list()
        # select predictions of every video (prob > threshold)
        for i in range(num_of_video):
            part = list()
            tmp_one_video = list(all_test.values())[i]
            for tmp in tmp_one_video:
                if mae_or_me == 2.0:
                    if float(tmp[4]) == 2.0:  # me
                        if float(tmp[3][:-1]) > k:
                            part.append(tmp)
                else:
                    if float(tmp[4]) == 1.0:  # me
                        if float(tmp[3][:-1]) > k:
                            part.append(tmp)
            # N: number of precictions of micro-expressions or macro-expressions
            N = N + len(part)

            if not part:
                part.append([tmp_one_video[0][0], '100000', '100000', '100000', '_'])
            part_tmp.append(part)
        part_pre = part_tmp

        # predictions: sorted by prob
        part_pre = [sorted(i, key=lambda x: int(float(x[1]))) for i in part_pre]

        # calculate iou between every prediction with GT
        for video_num, pre in enumerate(part_pre):
            video_name_list = list(set(ann_csv.video.values[:].tolist()))
            video_name_list.sort()

            # identify the current video
            video_name_last = part_pre[video_num][0][0]
            if dataset == 'cas(me)^2':
                video_name_part = 's' + video_name_last[:2]
                video_name = os.path.join(os.path.dirname(os.path.dirname(video_name_list[0])), video_name_part,
                                          video_name_last)
            elif dataset == 'samm_merge':
                video_name = os.path.join(video_name_list[0][:-4], str(video_name_last).zfill(3))
            elif dataset == 'cas(me)^3':
                video_name = os.path.join(video_name_list[0][:-6], str(video_name_last.split('_')[0]).zfill(3),
                                          str(video_name_last.split('_')[1]))
            else:
                video_name = os.path.join(video_name_list[0][:-6], str(video_name_last).zfill(3))
            # select startframes of current video
            video_ann_df = ann_csv[ann_csv.video == video_name]
            if mae_or_me == 2.0:
                video_ann_df = video_ann_df[video_ann_df.type_idx == 2.0]
            else:
                video_ann_df = video_ann_df[video_ann_df.type_idx == 1.0]
            if len(video_ann_df) == 0:
                continue
            act_start_video = video_ann_df['startFrame'].values[:]
            # select indexes of startframes of current video
            indexes = np.argsort(act_start_video)
            # labels and endframes are sorted by indexes from actual start frames
            act_end_video = video_ann_df['endFrame'].values[:]
            act_end_video = np.array(act_end_video)[indexes]
            # actual start frames are sorted by time series
            act_start_video.sort()

            pre = np.array(pre)
            pre_conf = []
            pre_start = []
            pre_end = []
            for ps in pre[:, 1].astype(float):
                if ps - int(ps) >= 0.5:
                    pre_start.append(int(ps) + 1)
                else:
                    pre_start.append(int(ps))

            for ps in pre[:, 2].astype(float):
                if ps - int(ps) >= 0.5:
                    pre_end.append(int(ps) + 1)
                else:
                    pre_end.append(int(ps))

            for conf in pre[:, 3].astype(float):
                pre_conf.append(conf)

            pre_start = np.array(pre_start)
            pre_end = np.array(pre_end)
            pre_conf = np.array(pre_conf)

            tp_sample_list = []
            for m in range(len(pre_start)):
                per_pre_start = pre_start[m]
                per_pre_end = pre_end[m]
                iou = (np.minimum(per_pre_end, act_end_video) - np.maximum(per_pre_start, act_start_video) + 1) / (
                        np.maximum(per_pre_end, act_end_video) - np.minimum(per_pre_start, act_start_video) + 1)
                max_iou = np.max(iou)
                max_index = np.argmax(iou)
                if max_iou >= iou_threshold:
                    tp_iou = iou[max_index]
                    assert tp_iou == max_iou
                    tp_sample = [video_name_last, per_pre_start, per_pre_end, act_start_video[max_index],
                                 act_end_video[max_index], tp_iou,  pre_conf[m], mae_or_me]
                    tp_sample_list.append(tp_sample)
            TP = TP + len(tp_sample_list)
            all_tp_sample_list += tp_sample_list
    return  TP, N, all_tp_sample_list

def main_threshold(path, dataset, annotation, me_start_threshold, mae_start_threshold, iou_threshold):
    results_txt = os.path.join(path, 'all_results.txt')
    detailed_txt = os.path.join(path, 'detailed_results.txt')
    proposals_txt = os.path.join(path, 'proposals_results.txt')
    if os.path.isfile(results_txt):
        os.remove(results_txt)
    if os.path.isfile(detailed_txt):
        os.remove(detailed_txt)
    if os.path.isfile(proposals_txt):
        os.remove(proposals_txt)

    files_tmp = os.listdir(path)
    files = sorted(files_tmp, key=lambda x: int(x[-2:]))
    ann_csv = pd.read_csv(annotation)
    test_path_temp = [os.path.join(path, i, 'test_detection') for i in files]
    txts = glob.glob(os.path.join(test_path_temp[0], '*.txt'))

    txts = [int(i.split('_')[-1].split('.')[0]) for i in txts]
    txts.sort()

    no_repeat_list = []
    for i in txts:
        if i not in no_repeat_list:
            no_repeat_list.append(i)
    txts = no_repeat_list

    if dataset == 'cas(me)^2' or dataset == 'cas(me)^2_merge':
        M1 = 300
        M2 = 57
    elif dataset == 'cas(me)^3':
        M1 = 2071
        M2 = 277
    else:
        M1 = 312
        M2 = 159

    with open(results_txt, 'a') as rf:
        for e in range(5, 30):
            best_recall, best_precision, best_overall, best_maek, best_mek = 0.00, 0.00, 0.00, 0.00, 0.00
            txt_index = txts[e]
            # all subjects in the same epoch
            test_path = [os.path.join(i, 'test_' + str(txt_index).zfill(2) + '.txt') for i in test_path_temp]
            # ME proposals
            T2 = list()
            N2_all = list()
            ME_K_list = list()
            for me_k_temp in range(me_start_threshold, 500, 1):
                me_k = 1.0 * me_k_temp / 1000
                ME_K_list.append(me_k)
                TP2, N2, _ = proposal_arrange(test_path, me_k, ann_csv, iou_threshold, 2.0)
                T2.append(TP2)
                N2_all.append(N2)
            # MAE proposals
            T1 = list()
            N1_all = list()
            MAE_K_list = list()
            for mae_k_temp in range(mae_start_threshold, 500, 1):
                mae_k = 1.0 * mae_k_temp / 1000
                MAE_K_list.append(mae_k)
                TP1, N1, _ = proposal_arrange(test_path, mae_k, ann_csv, iou_threshold, 1.0)
                T1.append(TP1)
                N1_all.append(N1)
            # overall proposals
            for tp1, all1, maek in zip(T1, N1_all, MAE_K_list):
                for tp2, all2, mek in zip(T2, N2_all, ME_K_list):
                    recall_all = 1.0 * (tp1 + tp2) / (M1 + M2)
                    F1_SCORE, precision_all = all_score(tp1, tp2, all1, all2, recall_all)
                    if F1_SCORE > best_overall:
                        best_overall = F1_SCORE
                        best_recall = recall_all
                        best_precision = precision_all
                        best_maek = maek
                        best_mek = mek
            print("EPOCH: %d  recall: %.4f, precision: %.4f, f1_score: %.4f, MAEK: %.3f, MEK: %.3f" %
                  (e+1, best_recall, best_precision, best_overall, best_maek, best_mek))
            results_list = [e + 1, best_recall, best_precision, best_overall, best_maek, best_mek]
            results_list_string = ' '.join(map(str, results_list))
            rf.write(results_list_string + '\n')

    # Detailed data on the best results
    print('----------------- Detailed results of ME, MaE and Overall -----------------')
    results_txt_line = []
    f1_score_list = []
    with open(results_txt, 'r') as file:
        for line in file:
            line_elements = line.strip().split()
            results_txt_line.append(line_elements)
            f1_score_list.append(float(line_elements[3]))
    max_f1_score = max(f1_score_list)
    max_index = f1_score_list.index(max_f1_score)
    max_mae_k = float(results_txt_line[max_index][-2])
    max_me_k = float(results_txt_line[max_index][-1])
    max_epoch = int(results_txt_line[max_index][0]) - 1
    print("Best epoch: %d"%(max_epoch))
    max_test_path = [os.path.join(i, 'test_' + str(max_epoch).zfill(2) + '.txt') for i in test_path_temp]
    # detailed data of me
    max_me_tp, max_me_n, max_me_sample = proposal_arrange(max_test_path, max_me_k, ann_csv, iou_threshold, 2.0)
    max_me_fp = max_me_n - max_me_tp
    max_me_fn = M2 - max_me_tp
    recall_me = max_me_tp / M2
    precision_me = max_me_tp / max_me_n if max_me_n != 0 else 0
    f1_score_me = 2 * (recall_me * precision_me) / (recall_me + precision_me) if (recall_me + precision_me) != 0 else 0
    print("ME_TP: %d, ME_FP: %d, ME_FN: %d, ME_N: %d, "
          "ME_recall: %.4f, ME_precision: %.4f, ME_f1_score: %.4f" %
          (max_me_tp, max_me_fp, max_me_fn, max_me_n, recall_me, precision_me, f1_score_me))
    # detailed data of mae
    max_mae_tp, max_mae_n, max_mae_sample = proposal_arrange(max_test_path, max_mae_k, ann_csv, iou_threshold, 1.0)
    max_mae_fp = max_mae_n - max_mae_tp
    max_mae_fn = M1 - max_mae_tp
    recall_mae = max_mae_tp / M1
    precision_mae = max_mae_tp / max_mae_n if max_mae_n != 0 else 0
    f1_score_mae = 2 * (recall_mae * precision_mae) / (recall_mae + precision_mae) if (recall_mae + precision_mae) != 0 else 0
    print("MaE_TP: %d, MaE_FP: %d, MaE_FN: %d, MaE_N: %d, "
          "MaE_recall: %.4f, MaE_precision: %.4f, MaE_f1_score: %.4f" %
          (max_mae_tp, max_mae_fp, max_mae_fn, max_mae_n, recall_mae, precision_mae, f1_score_mae))
    # detailed data of overall
    max_all_tp = max_me_tp + max_mae_tp
    max_all_fp = max_me_fp + max_mae_fp
    max_all_n = max_me_n + max_mae_n
    max_all_fn = max_me_fn + max_mae_fn
    recall_all = max_all_tp / (M1 + M2)
    precision_all = max_all_tp / max_all_n if max_all_n != 0 else 0
    f1_score_all = 2 * (recall_all * precision_all) / (recall_all + precision_all) if (recall_all + precision_all) != 0 else 0
    print("ALL_TP: %d, ALL_FP: %d, ALL_FN: %d, ALL_N: %d, "
          "ALL_recall: %.4f, ALL_precision: %.4f, ALL_f1_score: %.4f" %
          (max_all_tp, max_all_fp, max_all_fn, max_all_n, recall_all, precision_all, f1_score_all))
    # 保存具体结果之间的关系到txt文件
    detailed_lines = []
    detailed_lines.append("Best epoch: %d"%(max_epoch))
    detailed_lines.append("ME_TP: %d, ME_FP: %d, ME_FN: %d, ME_N: %d, "
          "ME_recall: %.4f, ME_precision: %.4f, ME_f1_score: %.4f" %
          (max_me_tp, max_me_fp, max_me_fn, max_me_n, recall_me, precision_me, f1_score_me))
    detailed_lines.append("MaE_TP: %d, MaE_FP: %d, MaE_FN: %d, MaE_N: %d, "
          "MaE_recall: %.4f, MaE_precision: %.4f, MaE_f1_score: %.4f" %
          (max_mae_tp, max_mae_fp, max_mae_fn, max_mae_n, recall_mae, precision_mae, f1_score_mae))
    detailed_lines.append("ALL_TP: %d, ALL_FP: %d, ALL_FN: %d, ALL_N: %d, "
          "ALL_recall: %.4f, ALL_precision: %.4f, ALL_f1_score: %.4f" %
          (max_all_tp, max_all_fp, max_all_fn, max_all_n, recall_all, precision_all, f1_score_all))
    with open(detailed_txt, 'w') as file:
        file.write("\n".join(detailed_lines))
    # 保存提案和GT之间的关系到txt文件
    proposals_lines = max_mae_sample + max_me_sample
    with open(proposals_txt, 'w') as file:
        for sublist in proposals_lines:
            file.write(" ".join(map(str, sublist)) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')

    # parser.add_argument('--path', type=str, default="/home/geek/spot/New_Code_2_3/IAENet_CNN/output/version1/")
    parser.add_argument('--path', type=str, default="/home/geek/spot/LXDNet/Lxdnet/output/old_samm_layer_regression_mlp_30/")
    # parser.add_argument('--ann', type=str, default="/home/geek/spot/New_Code_2_3/Annation_csv/casme2_spot.csv")
    parser.add_argument('--ann', type=str, default="/home/geek/spot/LXDNet/samm_annotation_merge_part_2000_L800_new2.csv")
    parser.add_argument('--dataset', type=str, default='samm')
    # parser.add_argument('--dataset', type=str, default='samm')
    parser.add_argument('--me_start_threshold', type=int, default=100)
    parser.add_argument('--mae_start_threshold', type=int, default=300)
    parser.add_argument('--iou_threshold', type=float, default=0.5)

    args = parser.parse_args()
    path = args.path
    dataset = args.dataset
    ann = args.ann
    me_start_threshold = args.me_start_threshold
    mae_start_threshold = args.mae_start_threshold
    iou_threshold = args.iou_threshold

    print(path)
    main_threshold(path, dataset, ann, me_start_threshold, mae_start_threshold, iou_threshold)