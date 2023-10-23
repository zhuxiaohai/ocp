CUDA_VISIBLE_DEVICES=3,4 python launch.py --nproc_per_node=2 main.py --distributed --num-gpus 2 \
--config-yml configs/tem_stat/predict_esm2_emb.yml \
--mode predict_async