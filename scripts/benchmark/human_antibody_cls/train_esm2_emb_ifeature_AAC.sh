CUDA_VISIBLE_DEVICES=0,2 python launch.py --nproc_per_node=2 main.py \
--distributed \
--num-gpus 2 \
--config-yml configs/tem_stat/train_esm2.yml \
--mode train