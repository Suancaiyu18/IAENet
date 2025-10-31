CUDA_VISIBLE_DEVICES=7 python /home/geek/spot/New_Code_2_3/IAENet_CNN/tools/main.py \
     --cfg "/home/geek/spot/New_Code_2_3/IAENet_CNN/experiments/cas2.yaml"\
     --dataset 'cas(me)^2' \

python /home/geek/spot/New_Code_2_3/IAENet_CNN/tools/F1_score_metrics.py \
    --path "/home/geek/spot/New_Code_2_3/IAENet_CNN/output/version_cas2_lamda_0.05_0/" \
    --ann "/home/geek/spot/IAENet_Plus/Annotations/casme2_annotation.csv" \
    --dataset 'cas(me)^2' \
    --me_start_threshold 0 \
    --mae_start_threshold 0