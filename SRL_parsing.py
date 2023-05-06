import json
import torch
import argparse
import src.pipeline
from tqdm import tqdm
from transformers import set_seed
import allennlp_models.pretrained
import os, logging
from util import init_logger


def disable_allennlp_logger():
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)


def chunk_list(seq_list, batch_size):
    out = []
    for start_idx in range(0, len(seq_list), batch_size):
        end_idx = min(start_idx + batch_size, len(seq_list))
        out.append(seq_list[start_idx: end_idx])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--data_path", type=str, help="path to your data")
    parser.add_argument("--model_path", type=str, default="structured-prediction-srl-bert",
                        help="path to your model or the name of your model")
    parser.add_argument("--output_dir", type=str,
                        help="path to your output dir")
    parser.add_argument("--output_file", type=str, default="srl.json")
    parser.add_argument("--log_file", type=str, default="log/srl.log")
    parser.add_argument("--batch_size", type=int, default=512)
    # parser.add_argument("--max_prompt_num", type=int, default=11)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("GPU: ", torch.cuda.is_available())
    device_id = 0 if torch.cuda.is_available() else None

    logger = init_logger(args, logging.INFO)

    disable_allennlp_logger()
    set_seed(args.seed)

    with open(args.data_path) as fin:
        data = [json.loads(line) for line in fin]

    sentence_list, id_list = [], []
    for d in data:
        for i in range(0, 4):
            sentence_list.append({"sentence": d["story"][i]})
            id_list.append(i)
        if args.debug and len(sentence_list) >= 4:
            break
    grouped_batches = chunk_list(sentence_list, args.batch_size)

    allensrl = src.pipeline.AllenSRLWrapper(
        allennlp_models.pretrained.load_predictor(args.model_path, cuda_device=device_id))

    codes = ["resemantic", "negation", "lexical", "quantifier",
                                                  "insert", "delete"]

    blank_list = []
    for sentence_batch in tqdm(grouped_batches, "parsing batches of sentences with SRL"):
        blank_result = allensrl.generate_blanks_via_srl(sentence_batch)
        blank_list.extend(blank_result)

    prompt_list = []
    for sent, blank in zip(sentence_list, blank_list):
        sent = sent["sentence"]
        prompt = allensrl.get_prompts(sent, codes, blank)
        print(len(prompt), end=" ")
        prompt_list.append(prompt)

    res_list = []
    cur_dict = {}
    for sent_id, prompt in zip(id_list, prompt_list):
        cur_dict[f"s{sent_id}"] = prompt
        if sent_id == 3:
            res_list.append(cur_dict)
            cur_dict = {}

    with open(os.path.join(args.output_dir, args.output_file), "w") as fout:
        for d in res_list:
            fout.write(json.dumps(d) + "\n")
