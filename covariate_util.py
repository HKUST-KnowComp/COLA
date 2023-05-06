import numpy as np
from typing import Optional
from dataclasses import dataclass, field, asdict
from transformers import Seq2SeqTrainingArguments


def format_args(args):
    args_as_dict = asdict(args)
    args_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in args_as_dict.items()}
    attrs_as_str = [f"{k}={v}," for k, v in sorted(args_as_dict.items())]
    return f"{args.__class__.__name__}\n({' '.join(attrs_as_str)})"


def decode_generation(pred_ids, input_ids, tokenizer, model_name_or_path):
    # Replace -100 as we can't decode them.
    pred_ids = [np.where(preds != -100, preds, tokenizer.pad_token_id) for preds in pred_ids]

    raw_preds = [tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    ) for preds in pred_ids]
    assert len(raw_preds) == len(input_ids)
    raw_preds = [[pred.strip() for pred in pred_list] for pred_list in raw_preds]

    if "gpt" in model_name_or_path:
        input_ids = [np.where(ipi != -100, ipi, tokenizer.pad_token_id) for ipi in input_ids]
        raw_inputs = [tokenizer.batch_decode(
            ipi, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ipi in input_ids]
        raw_inputs = [inp[0].strip() for inp in raw_inputs]
        raw_preds = [[pd[len(inp):].strip() for pd in pd_list] for pd_list, inp in zip(raw_preds, raw_inputs)]

    return raw_preds


def get_preprocess_function(data_args, tokenizer, model_name_or_path, joint=False):
    def preprocess_function(examples):
        # remove pairs where at least one record is None
        padding = "max_length" if data_args.pad_to_max_length else False
        inputs = []
        for i in range(len(examples[data_args.text_column])):
            if not joint:
                inputs.append(examples[data_args.text_column][i])
            else:
                last_idx = i - i % 4 + 4
                input_text = " ".join(examples[data_args.text_column][i: last_idx])
                inputs.append(input_text)

        # BART example: [event] Before that,
        if not joint:
            prompt = "{} Before that,"
        else:
            prompt = "{} Before all events, "

        # T5 example: [event] Before that, <extra_id_0>
        if "t5" in model_name_or_path:
            prompt += "<extra_id_0>"

        inputs = [prompt.format(i) for i in inputs]

        model_inputs = [tokenizer(ipt, padding=padding, truncation=True,
                                  return_tensors="pt") for ipt in inputs]
        return model_inputs

    return preprocess_function


@dataclass
class ExtendedSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    output_file: str = field(
        default="covariates.json",
        metadata={"help": "file that sampled covariates are written. It is concatenated with `output_dir` "
                          "(final path: output_dir/output_file)"}
    )
    log_file: str = field(
        default="log/covariates.log",
        metadata={"help": "log file. It is concatenated with `output_dir` "
                          "(final path: output_dir/log_file)"}
    )
    debug: bool = field(
        default=False,
        metadata={"help": "debug, less data"}
    )
    start_idx: int = field(
        default=-1,
        metadata={"help": "starting index of the dataset."}
    )
    end_idx: int = field(
        default=-1,
        metadata={"help": "ending index of the dataset"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    data_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=30,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=10,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    min_length: int = field(default=5,
                            metadata={"help": "minimum generation length"})
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_return_sequences: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of sequences you want your model to return for one example"
        },
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    do_sample: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "whether do sample during generation."
            )
        }
    )

    def __post_init__(self):
        if self.dataset_name is None and self.data_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.data_file is not None:
                extension = self.data_file.split(".")[-1]
                assert extension in ["csv", "json"], "`data_file` should be a csv or a json file."
