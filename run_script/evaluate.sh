cov_inter_path=YOUR_COV_INTER_TEMPORAL_SCORES_PATH
inter_outcome_path=YOUR_INTER_OUTCOME_TEMPORAL_SCORES_PATH
output_file=YOUR_OUTPUT_PATH
python evaluate.py --cov_inter_path ${cov_inter_path} --inter_outcome_path ${inter_outcome_path} \
--output_file ${output_file} --eps_start 0.0 --eps_end 0.02 --eps_count 21 \
--data_path ./COPES_data/COPES.json

