import os

import torch
import numpy as np
import ujson as json
from tqdm import tqdm
from datasets import Dataset
from util import init_logger, is_main_process, init_output_dir
from transformers import AutoTokenizer, set_seed
from transformers import AutoModelForCausalLM
from transformers import HfArgumentParser, AutoConfig
from covariate_util import ExtendedSeq2SeqTrainingArguments, DataTrainingArguments, ModelArguments
from covariate_util import get_preprocess_function, decode_generation, format_args


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,
                               ExtendedSeq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    print("initializing the output dir")
    if is_main_process(training_args.local_rank):
        init_output_dir(training_args)

    print("initializing the logger")
    log_level = training_args.get_process_log_level()
    with training_args.main_process_first(desc="getting logger"):
        logger = init_logger(training_args, log_level)

    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}, " +
        f"bf16 training: {training_args.bf16}"
    )

    if is_main_process(training_args.local_rank):
        logger.info(format_args(training_args))
        logger.info(format_args(data_args))
        logger.info(format_args(model_args))

    logger.info("unpack data")
    with open(data_args.data_file) as fin:
        data_list = [json.loads(line) for line in fin]
        if training_args.debug:
            data_list = data_list[: 1]
        if training_args.start_idx < 0:
            training_args.start_idx = None
        if training_args.end_idx < 0:
            training_args.end_idx = None
        if training_args.start_idx is not None or training_args.end_idx is not None:
            if training_args.start_idx is None:
                training_args.start_idx = 0
            if training_args.end_idx is None:
                training_args.end_idx = len(data_list)
            data_list = data_list[training_args.start_idx: training_args.end_idx]
            name, extension = training_args.output_file.split(".")
            name += f"_{training_args.start_idx}_{training_args.end_idx}"
            training_args.output_file = f"{name}.{extension}"
        unpacked_data_list = [{"text": d["story"][i]} for d in data_list for i in range(0, 4)]
        id_list = [i for d in data_list for i in range(0, 4)]
    print(len(unpacked_data_list))
    open(os.path.join(training_args.output_dir, training_args.output_file), "w").close()
    dataset = Dataset.from_list(unpacked_data_list)

    extension = data_args.data_file.split(".")[-1]
    column_names = dataset.column_names

    logger.info("loading model")
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if "gpt" in model_args.model_name_or_path and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    cur_preprocess_function = get_preprocess_function(data_args, tokenizer=tokenizer,
                                                      model_name_or_path=model_args.model_name_or_path, joint=True)

    if data_args.max_samples is not None:
        max_samples = min(len(dataset), data_args.max_samples)
        dataset = dataset.select(range(max_samples))

    logger.info("preprocessing data")
    dataset = cur_preprocess_function(dataset)

    logger.info("*** Predict process {} ***".format(training_args.local_rank))

    logger.info("local rank {} running on eval dataset".format(training_args.local_rank))

    gen_kwargs = {
        "num_return_sequences": data_args.num_beams,
        "num_beams": data_args.num_beams,
        "min_length": data_args.min_length,
        "do_sample": data_args.do_sample,
        "temperature": 0.9,
        "early_stopping": False
    }

    generated_id_list, input_id_list = [], []
    for data in tqdm(dataset):
        data = {key: value.to(device) for key, value in data.items()}
        pred_results = []
        for _ in range(data_args.num_return_sequences // data_args.num_beams):
            part_results = model.generate(data["input_ids"],
                                          attention_mask=data["attention_mask"],
                                          max_length=data_args.max_target_length + len(data["input_ids"][0]),
                                          **gen_kwargs)
            # part_sent = tokenizer.batch_decode(part_results, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            pred_results.append(part_results.cpu().numpy())
        pred_results = np.concatenate(pred_results)
        generated_id_list.append(pred_results)
        input_id_list.append(data["input_ids"].cpu().numpy())

    covariates = decode_generation(generated_id_list, input_id_list, tokenizer,
                                   model_args.model_name_or_path)

    res_list = []
    cur_cov_dict = {}
    for sent_id, cov in zip(id_list, covariates):
        cur_cov_dict[f"s{sent_id}"] = cov
        if sent_id == 3:
            res_list.append(cur_cov_dict)
            cur_cov_dict = {}

    with open(os.path.join(training_args.output_dir, training_args.output_file), "w") as fout:
        for d in res_list:
            fout.write(json.dumps(d) + "\n")
