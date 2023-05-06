import json
import torch
import random
import argparse
import os, logging
import src.pipeline
from tqdm import tqdm
from util import init_logger
from transformers import set_seed


def disable_allennlp_logger():
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('transformers.generation_utils').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)


def duplicate_list(my_list, length):
    idx, new_list = 0, []
    while len(new_list) < length:
        new_list.append(my_list[idx])
        idx = (idx + 1) % len(my_list)
    return new_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--data_path", type=str, help="path to your data")
    parser.add_argument("--model_path", type=str, default="uw-hai/polyjuice",
                        help="path to your model or the name of your model")
    parser.add_argument("--output_dir", type=str,
                        help="path to your output dir")
    parser.add_argument("--log_file", type=str, default="log/inter.log")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_file", type=str, default="inter.json")
    parser.add_argument("--max_prompt_num", type=int, default=10)

    # decoding arguments
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=40)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    args = parser.parse_args()

    print("GPU: ", torch.cuda.is_available())
    device_id = 0 if torch.cuda.is_available() else None

    logger = init_logger(args, logging.INFO)

    disable_allennlp_logger()
    set_seed(args.seed)

    data = []
    with open(args.data_path) as fin:
        for line in fin:
            data.append(json.loads(line))
            if args.debug and len(data) >= 4:
                break

    # intervention arguments
    gen_kwargs = {
        "max_new_tokens": args.max_length,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "do_sample": False,
        "temperature": 1,
    }

    cf_gen = src.pipeline.PJGenerator(model_path=args.model_path, device=device_id)

    res_list = []
    for d in tqdm(data, "generating data"):
        inter_dict = {}
        for i in range(0, 4):
            if len(d[f"s{i}"]) >= args.max_prompt_num + 1:
                chosen_prompt_list = random.sample(d[f"s{i}"], args.max_prompt_num + 1)
            else:
                print(len(d[f"s{i}"]), "not enough")
                chosen_prompt_list = duplicate_list(d[f"s{i}"], args.max_prompt_num + 1)
                # print(chosen_prompt_list)
            interventions = cf_gen(chosen_prompt_list, **gen_kwargs)
            # clean the same one
            origin_event = d[f"s{i}"][0].split(" <|perturb|>")[0].strip()
            if origin_event in interventions:
                interventions.remove(origin_event)
            # else:
            #     interventions = interventions[:args.max_inter_num]
            print(len(interventions), end=" ")
            saved_number = args.max_prompt_num * args.num_return_sequences
            if len(interventions) < saved_number:
                interventions = duplicate_list(interventions, saved_number)
            inter_dict[f"s{i}"] = interventions[: saved_number]
        res_list.append(inter_dict)

    with open(os.path.join(args.output_dir, args.output_file), "w") as fout:
        for r in res_list:
            fout.write(json.dumps(r) + "\n")
