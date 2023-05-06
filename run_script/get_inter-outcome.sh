model_path=bert-base-uncased
output_dir=YOUR_OUTPUT_DIR
data_path=./COPES_data/COPES.json
crop=0
inter_path=YOUR_INTERVENTION_FILE_PATH

python get_temporal_scores.py \
--data_path ${data_path} \
--inter_path ${inter_path} \
--model_path ${model_path} \
--output_dir ${output_dir} \
--temporal_type inter-outcome --cov_num 50 --inter_num 51 --batch_size 512 \
--crop ${crop}