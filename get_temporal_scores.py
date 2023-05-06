import os
import torch
import logging
import argparse
import numpy as np
import src.pipeline
from copy import deepcopy
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import set_seed
from temporal_score_util import unpacked_data, get_preprocess_function
from temporal_score_util import gather_cov_inter_data, gather_inter_outcome_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--data_path", type=str, help="path to your data")
    parser.add_argument("--cov_path", type=str, help="path to covariates")
    parser.add_argument("--inter_path", type=str, help="path to interventions")
    parser.add_argument("--model_path", type=str,
                        help="path to your model or the name of your model")
    parser.add_argument("--output_dir", type=str,
                        help="path to your output dir")
    parser.add_argument("--max_length", type=int, default=90,
                        help="max event pair")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4,
                        help="number of threads to preprocess data")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="overwrite the cache for processed data")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="inference batch size")
    parser.add_argument("--temporal_type", type=str, default="inter-outcome",
                        choices=["cov-inter", "inter-outcome"],
                        help="the temporal relation of which step you want to get")
    parser.add_argument("--cov_num", type=int, default=100)
    parser.add_argument("--inter_num", type=int, default=11)
    parser.add_argument("--debug", action="store_true", help="debugging mode ")
    parser.add_argument("--crop", type=int, default=1,
                        help="crop letters in the prompted output, for BERT-base/large your should use 0"
                             "for RoBERTa and DeBERTa, you should use 1")
    parser.add_argument("--top_k", type=int, default=5,
                        help="number of tokens to prompt")
    args = parser.parse_args()

    print(args)

    logging.getLogger().setLevel(logging.INFO)
    set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert torch.cuda.is_available()
    device_id = 0

    if args.temporal_type == "cov-inter":
        packed_event_pair_list = gather_cov_inter_data(cur_args=args)
    elif args.temporal_type == "inter-outcome":
        packed_event_pair_list = gather_inter_outcome_data(cur_args=args)
    else:
        raise ValueError("Wrong value of temporal_type: {}".format(args.temporal_type))

    # merge [event + intervention] and outcome
    unpacked_event_pair_list = unpacked_data(packed_event_pair_list, "text", "outcome")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    forward_preprocess_func = get_preprocess_function("text", "outcome", tokenizer,
                                                      tokenizer.mask_token, args.max_length)
    backward_preprocess_func = get_preprocess_function("outcome", "text", tokenizer,
                                                       tokenizer.mask_token, args.max_length)

    # print(unpacked_event_pair_list)
    print(len(unpacked_event_pair_list))
    forward_dataset = Dataset.from_list(unpacked_event_pair_list)
    backward_dataset = deepcopy(forward_dataset)
    column_names = forward_dataset.column_names

    forward_dataset = forward_dataset.map(
        forward_preprocess_func,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on forward dataset",
    )

    backward_dataset = backward_dataset.map(
        backward_preprocess_func,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on backward dataset",
    )

    temporal_predictor = src.pipeline.TempPredictor(
        model=AutoModelForMaskedLM.from_pretrained(args.model_path),
        tokenizer=tokenizer,
        device=device_id)

    forward_tuple = temporal_predictor.batch_predict(forward_dataset, batch_size=args.batch_size, top_k=args.top_k)
    backward_tuple = temporal_predictor.batch_predict(backward_dataset, batch_size=args.batch_size, top_k=args.top_k)

    prob = temporal_predictor.postprocess_prob(forward_tuple, backward_tuple,
                                               device=device_id, crop=args.crop)

    # reshape the matrix
    if args.temporal_type == "cov-inter":
        prob = prob.reshape((-1, args.cov_num, args.inter_num, prob.shape[-1]))
    elif args.temporal_type == "inter-outcome":
        prob = prob.reshape((-1, args.inter_num, 1, prob.shape[-1]))
    else:
        raise ValueError(f"Temporal type {args.temporal_type} doesn't exist.")

    np.save(os.path.join(args.output_dir, f"scores_{args.temporal_type}.json"), prob)
