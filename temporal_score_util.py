import json


def _sent_lowercase(s):
    try:
        return s[0].lower() + s[1:]
    except:
        return s


def _remove_punct(s):
    try:
        return s[:-1]
    except:
        return s


def get_input_pair(text_event_list, outcome_event_list):
    input_list = []
    for text in text_event_list:
        text = _remove_punct(text)
        for outcome in outcome_event_list:
            outcome = _sent_lowercase(outcome)
            input_list.append({"text": text, "outcome": outcome})
    return input_list


def unpacked_data(example_list, text_column, outcome_column):
    # remove pairs where at least one record is None
    event_pair_list = []
    for cur_example in example_list:
        text_event_list = cur_example[text_column]
        outcome_event_list = cur_example[outcome_column]
        event_pair_list.extend(get_input_pair(text_event_list, outcome_event_list))

    return event_pair_list


def get_preprocess_function(text_column, outcome_column, tokenizer, mask_token, max_length):
    def preprocess_function(examples):
        prompt_template = "{} " + mask_token + " {}"
        inputs = examples[text_column]
        targets = examples[outcome_column]
        # remove pairs where at least one record is None
        input_list = [prompt_template.format(i, t) for i, t in zip(inputs, targets)]
        model_inputs = tokenizer(input_list, max_length=max_length, padding=False, truncation=True)

        return model_inputs

    return preprocess_function


def gather_cov_inter_data(cur_args):
    with open(cur_args.data_path) as fin:
        origin_data = [json.loads(data_line) for data_line in fin]
    with open(cur_args.cov_path) as fin:
        cov_data = [json.loads(data_line) for data_line in fin]
    with open(cur_args.inter_path) as fin:
        inter_data = [json.loads(data_line) for data_line in fin]

    if cur_args.debug:
        origin_data, cov_data, inter_data = origin_data[:1], cov_data[:1], inter_data[:1]

    # merge covariates and [event + intervention]
    cov_inter_list = []
    for event_seq, cov_seq, inter_seq in zip(origin_data, cov_data, inter_data):
        for i in range(0, 4):
            cov_event = cov_seq[f"s{i}"]
            inter_event = [event_seq["story"][i]] + inter_seq[f"s{i}"]
            cov_inter_pair = {"text": cov_event,
                              "outcome": inter_event}
            cov_inter_list.append(cov_inter_pair)
    print(len(cov_inter_list[0]["text"]), len(cov_inter_list[0]["outcome"]))
    return cov_inter_list


def gather_inter_outcome_data(cur_args):
    with open(cur_args.data_path) as fin:
        origin_data = [json.loads(data_line) for data_line in fin]
    with open(cur_args.inter_path) as fin:
        inter_data = [json.loads(data_line) for data_line in fin]

    if cur_args.debug:
        origin_data, inter_data = origin_data[:1], inter_data[:1]

    # merge covariates and [event + intervention]
    inter_outcome_list = []
    for event_seq, inter_seq in zip(origin_data, inter_data):
        for i in range(0, 4):
            inter_event = [event_seq["story"][i]] + inter_seq[f"s{i}"]
            outcome_event = [event_seq["story"][4]]
            inter_outcome_pair = {"text": inter_event,
                                  "outcome": outcome_event}
            inter_outcome_list.append(inter_outcome_pair)
    print(len(inter_outcome_list[0]["text"]), len(inter_outcome_list[0]["outcome"]))
    return inter_outcome_list
