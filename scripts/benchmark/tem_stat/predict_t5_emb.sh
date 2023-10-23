CUDA_VISIBLE_DEVICES=0,2 python launch.py --nproc_per_node=2 main.py --distributed --num-gpus 2 \
--config-yml configs/tem_stat/predict_t5_emb.yml \
--mode predict_async