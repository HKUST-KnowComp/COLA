lr=1e-5
epoch=10
train_bs=512
eval_bs=512
model_name=bert-base-uncased
gpu_num=4
data_dir=YOUR_DIR
output_dir=YOUR_DIR
python -m torch.distributed.launch --nproc_per_node ${gpu_num} mlm.py \
--output_dir ${output_dir} --do_train \
--per_device_train_batch_size ${train_bs} --per_device_eval_batch_size ${eval_bs} \
--eval_accumulation_steps 8 --evaluation_strategy epoch --report_to none \
--learning_rate ${lr} --num_train_epochs ${epoch} --logging_strategy no --group_by_length \
--train_file ${data_dir}/train.json \
--validation_file ${data_dir}/valid.json \
--test_file ${data_dir}/test.json \
--save_strategy epoch --save_total_limit 1 \
--load_best_model_at_end --metric_for_best_model eval_accuracy --greater_is_better True \
--model_name_or_path ${model_name} --do_train --do_eval

