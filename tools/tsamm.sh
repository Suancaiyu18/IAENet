CUDA_VISIBLE_DEVICES=6 python /home/geek/spot/New_Code_2_3/IAENet_CNN/tools/main.py \
     --cfg "/home/geek/spot/New_Code_2_3/IAENet_CNN/experiments/samm.yaml"\
     --dataset 'samm' \

python /home/geek/spot/New_Code_2_3/IAENet_CNN/tools/F1_score_metrics.py \
    --path "/home/geek/spot/New_Code_2_3/IAENet_CNN/output/version_samm6/" \
    --ann "/home/geek/spot/New_Code_2_3/Annation_csv/samm_spot.csv" \
    --dataset 'samm' \
    --me_start_threshold 0 \
    --mae_start_threshold 0