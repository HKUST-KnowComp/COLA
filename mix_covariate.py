import json
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--context_path", type=str, help="data path to get context",
                        default="")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--mix_ground_truth", action="store_true",
                        help="mix ground truth")
    args = parser.parse_args()

    with open(args.input_path) as fin:
        data_list = [json.loads(line) for line in fin]

    with open(args.context_path) as fin:
        context_list = [json.loads(line) for line in fin]

    covariates_count = len(data_list[0]["s0"])
    covariates_part = covariates_count // 4
    for data, cont in tqdm(zip(data_list, context_list), total=len(data_list)):
        for i in range(1, 4):
            new_covariates_list = data[f"s{i}"][:-i * covariates_part]
            for j in range(0, i):
                new_covariates_list += data[f"s{j}"][: covariates_part]
            if args.mix_ground_truth:
                new_covariates_list = new_covariates_list[:-i] + cont["story"][:i]
            data[f"s{i}"] = new_covariates_list

    with open(args.output_path, "w") as fout:
        for data in data_list:
            fout.write(json.dumps(data) + "\n")