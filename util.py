import os
import shutil
import logging
import datasets
import transformers

from dataclasses import asdict
from transformers import MODEL_FOR_MASKED_LM_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def is_main_process(local_rank):
    return local_rank == 0 or local_rank == -1


def init_output_dir(training_args):
    if training_args.do_train and os.path.exists(training_args.output_dir):
        if os.path.exists(os.path.join(training_args.output_dir, "checkpoint_finish")) > 0:
            raise ValueError(
                "training process in dir {} is finished, plz clear it manually.".format(training_args.output_dir))
        shutil.rmtree(training_args.output_dir, ignore_errors=True)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    file_name = getattr(training_args, "log_file", "train.log")

    log_dir_path = os.path.join(training_args.output_dir, "log")
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    os.system("touch {}".format(os.path.join(training_args.output_dir, file_name)))


def init_logger(training_args, log_level):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # init a formatter to add date information
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # init a file handler and a stream handler
    file_name = getattr(training_args, "log_file", "train.log")
    fh = logging.FileHandler(os.path.join(training_args.output_dir, file_name), encoding="utf-8", mode="a")
    fh.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    # set formatter to handlers
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add those handlers to the root logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # the logger level of huggingface packages
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.enable_propagation()

    return logger


def format_args(args):
    args_as_dict = asdict(args)
    args_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in args_as_dict.items()}
    attrs_as_str = [f"{k}={v}," for k, v in sorted(args_as_dict.items())]
    return f"{args.__class__.__name__}\n({' '.join(attrs_as_str)})"
