CUDA_VISIBLE_DEVICES=2,3 python launch.py --nproc_per_node=2 main.py --distributed --num-gpus 2 \
--config-yml configs/human_antibody_cls/predict_esm2_emb.yml \
--mode predict_async