python scripts/preprocess_csv.py \
--idx idx \
--labels stability_at_40,stability_at_45,stability_at_50,stability_at_55,stability_at_60,stability_at_65 \
--sequence_field sequence \
--dataset_config configs/tem_stat/csv_2_lmdb.yml \
--num-workers 8