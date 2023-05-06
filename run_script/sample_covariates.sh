model_name=EleutherAI/gpt-j-6b
data_file=./COPES_data/COPES.json
output_dir=YOUR_DIR
output_file_name=YOUR_FILE_NAME

CUDA_VISIBLE_DEVICES=0 python sample_covariates.py \
--max_target_length 15 \
--model_name_or_path ${model_name} --text_column text \
--num_beams 10 --num_return_sequences 50 --report_to none \
--per_device_eval_batch_size 1 --eval_accumulation_steps 1 \
--data_file ${data_file} \
--output_dir ${output_dir} --output_file ${output_file} \
--do_sample True