python scripts/preprocess_pkl.py \
--idx id \
--labels no-polyreactivity \
--dense_fields t5_embedding,esm2_embedding,AAC \
--dataset_config configs/polyreactivity/multimodal_t5_emb_esm2_emb_ifeature_AAC.yml \
--num-workers 8