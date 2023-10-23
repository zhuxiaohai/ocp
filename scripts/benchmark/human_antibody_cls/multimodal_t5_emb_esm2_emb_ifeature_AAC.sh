python scripts/preprocess_pkl.py \
--idx id \
--labels label \
--dense_fields t5_embedding,esm2_embedding,AAC \
--dataset_config configs/human_antibody_cls/multimodal_t5_emb_esm2_emb_ifeature_AAC.yml \
--num-workers 8