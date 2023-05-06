#!/usr/bin/env python
# coding: utf-8
import os
import copy
import json
import argparse
import itertools
import random

import numpy as np
from time import time
from tqdm import tqdm
import src.metric_utils


class Result:
    def __init__(self, config_str, performance, pred_label_list):
        self_dict = copy.deepcopy(config_str)
        for key, value in performance.items():
            self_dict[key] = value
        self.summary = self_dict
        self.pred_label_list = pred_label_list

    def __str__(self):
        return json.dumps(self.summary)


def get_config_dict(dm, nm, pn, co, rn, eps):
    return {"D": dm, "S": nm, "Q": pn, "C": co, "E": rn, "eps": eps}


def get_f1(causal_label_list, ground_truth_list):
    shared_positive_list = causal_label_list * ground_truth_list
    shared_count = np.sum(shared_positive_list)
    p_count = np.sum(causal_label_list)
    r_count = np.sum(ground_truth_list)

    precision = shared_count / p_count if p_count != 0 else 0
    recall = shared_count / r_count if r_count != 0 else 0
    f1 = 2 * recall * precision / (recall + precision)

    return precision, recall, f1


def get_macro_f1(causal_label_list, ground_truth_list):
    p1, r1, f1 = get_f1(causal_label_list, ground_truth_list)

    reversed_causal_label_list = 1 - causal_label_list
    reversed_ground_truth_list = 1 - ground_truth_list
    p0, r0, f0 = get_f1(reversed_causal_label_list, reversed_ground_truth_list)

    p = (p0 + p1) / 2
    r = (r0 + r1) / 2
    f = (f0 + f1) / 2
    return p, r, f, f1


def get_acc(causal_label_list, ground_truth_list):
    shared_positive_list = causal_label_list == ground_truth_list
    shared_count = np.sum(shared_positive_list)
    return shared_count / len(shared_positive_list)


def pack_list(my_list, pl=4):
    packed_list = [my_list[my_idx: my_idx + pl] for my_idx in range(0, len(my_list), pl)]
    if isinstance(my_list, np.ndarray):
        packed_list = np.array(packed_list)
    return packed_list


def get_scores(causal_label_list, ground_truth_list, split_idx_list=None):
    causal_label_list = np.array(causal_label_list)
    ground_truth_list = np.array(ground_truth_list)
    if split_idx_list is None:
        res = {}
        res["acc"] = get_acc(causal_label_list, ground_truth_list)
        res["ma-p"], res["ma-r"], res["ma-f1"], res["f1"] = get_macro_f1(causal_label_list, ground_truth_list)
    else:
        causal_label_list, ground_truth_list = pack_list(causal_label_list), pack_list(ground_truth_list)
        valid_idx_list, test_idx_list = split_idx_list["valid"], split_idx_list["test"]
        eval_label_list = causal_label_list[valid_idx_list], causal_label_list[test_idx_list]
        eval_truth_list = ground_truth_list[valid_idx_list], ground_truth_list[test_idx_list]

        eval_label_list = [split_label_list.reshape(-1) for split_label_list in eval_label_list]
        eval_truth_list = [split_truth_list.reshape(-1) for split_truth_list in eval_truth_list]
        res = {}
        for name, label_list, truth_list in zip(["valid", "test"], eval_label_list, eval_truth_list):
            res[f"{name}_acc"] = get_acc(label_list, truth_list)
            res[f"{name}_ma-p"], res[f"{name}_ma-r"], \
            res[f"{name}_ma-f1"], res[f"{name}_f1"] = get_macro_f1(label_list, truth_list)

    return res


def get_metric(causal_score_list, ground_truth_list, config_dict, packed_data, split_idx_list=None):
    causal_label_list = []
    for group_idx, data in zip(range(0, len(causal_score_list), 4), packed_data):
        group_score = causal_score_list[group_idx: group_idx + 4]
        k = len(data["cause_idx"])
        if k > 0:
            top_indices = np.argpartition(group_score, -k)[-k:]
            cur_pred_label = [int(i in top_indices) for i in range(4)]
        else:
            cur_pred_label = [0] * 4
        causal_label_list.extend(cur_pred_label)

    performance = get_scores(causal_label_list, ground_truth_list, split_idx_list)
    res = Result(config_dict, performance, causal_label_list)
    return res


def get_latex_from_dict(metric_dict, split_name="all", contain_null=True):
    if split_name == "all":
        ordered_name_list = ["valid_acc", "valid_f1", "valid_ma-f1", "null",
                             "test_acc", "test_f1", "test_ma-f1", "null"]
    elif split_name == "valid":
        ordered_name_list = ["valid_acc", "valid_f1", "valid_ma-f1", "null"]
    elif split_name == "test":
        ordered_name_list = ["test_acc", "test_f1", "test_ma-f1", "null"]
    else:
        raise ValueError("Wrong split name")

    if not contain_null:
        ordered_name_list = [metric for metric in ordered_name_list if metric != "null"]

    latex_format_str = ""
    for cur_metric in ordered_name_list:
        if cur_metric != "null":
            score = metric_dict[cur_metric]
            score *= 100
            score = round(score, 2)
            str_score = "{:.2f}".format(score)
        else:
            str_score = "-"
        latex_format_str += "&" + str_score
    return latex_format_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--cov_inter_path", type=str)
    parser.add_argument("--inter_outcome_path", type=str)
    parser.add_argument("--choice_num", type=int, default=4)
    parser.add_argument("--cov_num", type=int, default=50)
    parser.add_argument("--inter_num", type=int, default=50)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--eps_start", default=0.00, type=float)
    parser.add_argument("--eps_end", default=0.02, type=float)
    parser.add_argument("--eps_count", default=21, type=int)
    parser.add_argument("--split_idx_file", type=str)
    parser.add_argument("--print", action="store_true", help="print performance")
    parser.add_argument("--order", default=2, type=int,
                        help="the order of computer distance")
    parser.add_argument("--split_name", default="valid", type=str)
    parser.add_argument("--second_order_key", default="all", type=str)
    parser.add_argument("--perf_split_name", default="all", type=str)

    args = parser.parse_args()

    print(args)
    with open(args.data_path) as fin:
        packed_data = [json.loads(line) for line in fin]

    if args.split_idx_file is not None:
        with open(args.split_idx_file) as fin:
            split_idx_list = json.load(fin)
    else:
        split_idx_list = None

    ground_truth_list = src.metric_utils.unpack_label(packed_data)
    cov_inter = np.load(args.cov_inter_path)
    inter_outcome = np.load(args.inter_outcome_path)
    inter_outcome = inter_outcome.squeeze()

    # check choice number
    unpacked_size, packed_size = cov_inter.shape[0], len(packed_data)
    assert unpacked_size == inter_outcome.shape[0]


    assert packed_size * args.choice_num == unpacked_size

    config_list = list(itertools.product(
        # D * S * Q
        [(False, True, True), (False, False, True),
         (False, False, False), (False, True, False),
         (True, False, False)
         ],
        # C * E
        [(False, True), (False, False), (True, False)],
    ))
    eps_start, eps_end, eps_number = args.eps_start, args.eps_end, args.eps_count
    eps_list = np.linspace(eps_start, eps_end, eps_number)

    performance_list = []

    idx = 0
    start_time = time()
    from copy import deepcopy
    copied_cov_inter, copied_inter_outcome = deepcopy(cov_inter), deepcopy(inter_outcome)
    for (dm, nm, pn), (co, rn) in config_list if args.print else tqdm(config_list):
        for eps in eps_list:
            causal_score_list = src.metric_utils.test_eps_copa(
                cov_inter, inter_outcome, args.cov_num, args.inter_num,
                direct_match=dm, normalization=nm, pxa_all_norm=pn,
                use_cooccur=co, res_norm=rn,
                ord=args.order, eps=eps)
            config_dict = get_config_dict(dm, nm, pn, co, rn, eps)
            max_res = get_metric(causal_score_list, ground_truth_list,
                                 config_dict, packed_data, split_idx_list)
            performance_list.append(max_res)
            if args.print:
                print(idx, performance_list[-1])
            idx += 1
    end_time = time()

    # sort by performance here
    if split_idx_list is None:
        performance_list = sorted(performance_list, key=lambda x: x.summary["f1"], reverse=True)
    else:
        def split_sum(score_dict, split_name):
            split_name_list = ["valid", "test"] if split_name == "all" else [split_name]
            score_sum = 0
            for cur_sn in split_name_list:
                score_sum += score_dict[f"{cur_sn}_f1"]
                score_sum += score_dict[f"{cur_sn}_ma-f1"]
                score_sum += score_dict[f"{cur_sn}_acc"]
            return score_sum

        if args.second_order_key is not None:
            performance_list = sorted(performance_list,
                                      key=lambda x: (split_sum(x.summary, args.split_name),
                                                     split_sum(x.summary, args.second_order_key)),
                                      reverse=True)
        else:
            performance_list = sorted(performance_list,
                                      key=lambda x: split_sum(x.summary, args.split_name),
                                      reverse=True)
    for trial in performance_list[: 10]:
        print(trial)
        print(get_latex_from_dict(trial.summary, args.perf_split_name))

    print("Used {}s to search the parameters".format(end_time - start_time))

    args.performance_file = os.path.join(args.output_file,
                                         "{}_{}_{}-{}-{}.json".format(args.split_name, args.order,
                                                                      eps_start, eps_end, eps_number))
    with open(args.performance_file, "w") as fout:
        for line in performance_list:
            fout.write(str(line) + "\n")
            latex_format = get_latex_from_dict(line.summary, args.perf_split_name)
            fout.write(latex_format + "\n")

    args.label_file = os.path.join(args.output_file, f"best_label.json")
    with open(args.label_file, "w") as fout:
        best_label_list = performance_list[0].pred_label_list
        for l in best_label_list:
            fout.write(str(l) + "\n")
