out_put_root=/shared/3/projects/hiatus/sadiri-mask/supcon_roberta_large_10_downsample/

top_k=200 # top_k frequent tokens
per=0 # percentage 
rate=0.0

seed=1234
CUDA="1"


run_name=top-$top_k-rate-$per-$seed-SupConLoss-cluster
# run_name=no-mask-42

out_put_dir=$out_put_root$run_name

CUDA_VISIBLE_DEVICES=$CUDA python main.py \
        --wandb \
        --train \
        --validate \
        --run_name $run_name \
        --out_dir $out_put_dir \
        --pretrained_model roberta-large \
        --project_name content-masking \
        --tokenizer roberta-large \
        --train_data /shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/down_10/train.jsonl \
        --dev_data /shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/down_10/dev \
        --learning_rate 0.00001 \
        --batch_size 128 \
        --epochs=5 \
        --max_length=512 \
        --grad_acc 1 \
        --gradient_checkpointing true \
        --saving_step 100 \
        --mask $rate \
        --seed $seed \
        --corpus /shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/down_10 \
        --top $top_k \
        --loss InfoNCE \
        --cluster

# CUDA_VISIBLE_DEVICES=3 python main.py \

        # --batch_size 128 \
        # --batch_size 256 \
        # --pretrained_model /shared/3/projects/hiatus/models/style-mlm/roberta-base-64/last/ \



# CUDA_VISIBLE_DEVICES="3,4" v \
#         main.py \
#         --train \
#         --validate \
#         --wandb \
#         --project_name Hiatus-k-hard \
#         --run_name NEW-1027-baselines-reddit-flan-t5-base \
#         --train_data /shared/3/projects/hiatus/hrs_hua/reddit/train \
#         --dev_data /shared/3/projects/hiatus/hrs_hua/reddit/dev \
#         --test_data /shared/3/projects/hiatus/hrs_hua/reddit/test \
#         --out_dir /shared/3/projects/hiatus/hrs_hua_ckpts/baselines_reddit_flan_t5_base \
#         --tokenizer google/flan-t5-base \
#         --pretrained_model google/flan-t5-base \
#         --batch_size 16 \
#         --learning_rate 1e-6 \
#         --epochs 10 \
#         --max_length=512 \
#         --saving_step=1000 \
#         --grad_acc 2 \


#         # --multivector \
#         # --decoder
#         # --train_data /shared/3/projects/hiatus/supreme_court/train \
#         # --dev_data /shared/3/projects/hiatus/supreme_court/dev \
#         # --test_data /shared/3/projects/hiatus/supreme_court/test \




# CUDA_VISIBLE_DEVICES="3,4" accelerate launch \
#         --num_processes=2 \
#         main.py \
#         --wandb \
#         --train \
#         --validate \
#         --project_name Hiatus-k-hard-roberta-large \
#         --run_name NEW-1027-baselines-reddit-roberta-large \
#         --train_data /shared/3/projects/hiatus/hrs_hua/reddit/train \
#         --dev_data /shared/3/projects/hiatus/hrs_hua/reddit/dev \
#         --test_data /shared/3/projects/hiatus/hrs_hua/reddit/test \
#         --out_dir /shared/3/projects/hiatus/hrs_hua_ckpts/baselines_reddit_roberta-large \
#         --tokenizer roberta-large \
#         --pretrained_model roberta-large \
#         --batch_size 8 \
#         --learning_rate 0.00001 \
#         --epochs 10 \
#         --max_length=512 \
#         --saving_step=1000 \
#         --grad_acc 2 \

#         # --loss contrastive \
#         # --sparse
#         # --gradient_checkpointing true 




# # Kenan
# accelerate launch --num_processes=3 \
#                 main.py \
#                 --wandb \
#                 --train \
#                 --validate \
#                 --gradient_checkpointing true \
#                 --run_name roberta-large-baseline \
#                 --out_dir /shared/3/projects/hiatus/models/TA1/ \
#                 --pretrained_model roberta-large \
#                 --tokenizer roberta-large \
#                 --train_data /shared/3/projects/hiatus/pretraining/data/train_long \
#                 --dev_data /shared/3/projects/hiatus/pretraining/data/dev_long \
#                 --learning_rate 0.00001 \
#                 --batch_size 12 \
#                 --max_length=512 \
#                 --evaluate \
#                 --grad_acc 4




