CUDA_VISIBLE_DEVICES=6,7 python launch.py --nproc_per_node=2 main.py --distributed --num-gpus 2 \
--config-yml configs/solubility/predict_esm2_emb.yml \
--mode predict_async