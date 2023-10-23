python scripts/preprocess_pkl.py \
--idx id \
--labels stability_at_40,stability_at_45,stability_at_50,stability_at_55,stability_at_60,stability_at_65 \
--dense_fields t5_embedding,esm2_embedding,AAC \
--dataset_config configs/tem_stat/multimodal_t5_emb_esm2_emb_ifeature_AAC.yml \
--num-workers 8