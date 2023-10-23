CUDA_VISIBLE_DEVICES=0,1 python launch.py --nproc_per_node=2 main.py \
--distributed \
--num-gpus 2 \
--config-yml configs/solubility/predict.yml \
--mode predict \
--checkpoint checkpoints/2023-10-10-13-54-08-solubility_t5_emb_train/best_checkpoint.pt