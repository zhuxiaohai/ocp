CUDA_VISIBLE_DEVICES=3,4 python launch.py --nproc_per_node=2 main.py \
--distributed \
--num-gpus 2 \
--config-yml configs/tem_stat/predict.yml \
--mode predict \
--checkpoint checkpoints/2023-10-09-23-00-16-tem_stat_t5_emb_train/best_checkpoint.pt