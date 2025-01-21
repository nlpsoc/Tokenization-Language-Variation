import argparse
import itertools
import os

from styletokenizer.glue import GLUE_TASKS, GLUE_TEXTFLINT_TASKS
from styletokenizer.utility.custom_logger import log_and_flush
from eval_tokenizer import calc_renyi_efficency_from_generator, calc_seq_len_from_generator, tok_generator, \
    _get_vocabsize_and_dist, calc_sim_renyi_efficiency_from_generator
from run_glue import task_to_keys as glue_task_to_keys
from styletokenizer.utility.env_variables import set_cache
from styletokenizer.tokenizer import TOKENIZER_PATHS
from styletokenizer.utility.datasets_helper import (load_data, VARIETIES_DEV_DICT, VARIETIES_to_keys, VARIETIES_TASKS,
                                                    VALUE_PATHS)

set_cache()

from styletokenizer.fitting_corpora import CORPORA_TWITTER, CORPORA_WIKIPEDIA, CORPORA_MIXED
from styletokenizer.utility.preptraining_corpora import CORPORA_WEBBOOK

FITTING_CORPORA = [CORPORA_TWITTER, CORPORA_WIKIPEDIA, CORPORA_MIXED, CORPORA_WEBBOOK]


def main(output_path=None):
    # task_name_or_hfpath = "mnli"
    # create a result dataframe with
    # task_name, tokenizer_path, renyi_eff_2.5, renyi_eff_3.0, avg_seq_len
    result_dict = {
        "task_name": [],
        "tokenizer_path": [],
        "renyi_eff_2.5": [],
        "renyi_sim_eff_2.5": [],
        "renyi_eff_3.0": [],
        "avg_seq_len": [],
        "vocab_size": []
    }
    for task_name_or_hfpath in (VARIETIES_TASKS + FITTING_CORPORA + GLUE_TASKS + GLUE_TEXTFLINT_TASKS):
        split = None
        csv_file = False
        if task_name_or_hfpath in VARIETIES_DEV_DICT.keys():
            task = task_name_or_hfpath
            task_name_or_hfpath = VARIETIES_DEV_DICT[task_name_or_hfpath]
            task_to_keys = VARIETIES_to_keys
            if task != "sadiri":
                csv_file = True
        elif task_name_or_hfpath in GLUE_TEXTFLINT_TASKS:
            task = task_name_or_hfpath
            task_to_keys = glue_task_to_keys
            split = "dev"
        elif task_name_or_hfpath in FITTING_CORPORA:
            task_to_keys = {"mixed": ["text"], "twitter": ["text"], "wikipedia": ["text"], "webbook": ["text"],
                            "pubmed": ["text"]}
            task = os.path.basename(os.path.normpath(task_name_or_hfpath))  # doesn't really matter which one
            split = "test"
        else:
            task = os.path.basename(os.path.normpath(task_name_or_hfpath))
            task_to_keys = glue_task_to_keys
        if csv_file:
            raw_datasets = load_data(csv_file=task_name_or_hfpath, split=split)
        else:
            raw_datasets = load_data(task_name_or_hfpath, split=split)
        try:
            eval_dataset = raw_datasets["validation_matched" if task == "mnli" else "validation"]
        except KeyError:  # some of the datasets are not provided in split form
            eval_dataset = raw_datasets
        sentence_keys = task_to_keys[task]
        text_generator = (" ".join(example[text_key] for text_key in sentence_keys if text_key is not None)
                         for example in eval_dataset)
        for tokenizer_group in TOKENIZER_PATHS:
            # get the smallest vocab size
            smallest_vocab_size = float("inf")
            for tokenizer_path in tokenizer_group:
                text_generator, t_gen1 = itertools.tee(text_generator, 2)
                tok_gen = tok_generator(t_gen1, tokenizer_path)
                vocab_size = _get_vocabsize_and_dist(tok_gen)[0]
                if vocab_size < smallest_vocab_size:
                    smallest_vocab_size = vocab_size
                result_dict["vocab_size"].append(vocab_size)
            log_and_flush(f"Simulated vocab size: {smallest_vocab_size} for group {tokenizer_group}")
            for tokenizer_path in tokenizer_group:
                out_path = f"{tokenizer_path}/intrinsic/{task}"
                text_generator, t_gen1, t_gen2, t_gen3, t_gen4 = itertools.tee(text_generator, 5)
                log_and_flush(f"\n{task_name_or_hfpath} - {tokenizer_path}")
                renyi_25 = calc_renyi_efficency_from_generator(t_gen1, tokenizer_path, power=2.5)
                log_and_flush(f"Renyi Efficency (2.5): "
                              f"{renyi_25}")
                sim_renyi_25 = calc_sim_renyi_efficiency_from_generator(t_gen4, tokenizer_path, smallest_vocab_size,
                                                                        power=2.5)
                log_and_flush(f"Sim Renyi Efficency (2.5): "
                                f"{sim_renyi_25}")
                renyi_30 = calc_renyi_efficency_from_generator(t_gen2, tokenizer_path, power=3.0)
                log_and_flush(f"Renyi Efficency (3.0): "
                              f"{renyi_30}")
                seq_len = calc_seq_len_from_generator(t_gen3, tokenizer_path)
                log_and_flush(f"Avg Seq Len: {seq_len}")
                result_dict["task_name"].append(task)
                result_dict["tokenizer_path"].append(tokenizer_path)
                result_dict["renyi_eff_2.5"].append(renyi_25)
                result_dict["renyi_sim_eff_2.5"].append(sim_renyi_25)
                result_dict["renyi_eff_3.0"].append(renyi_30)
                result_dict["avg_seq_len"].append(seq_len)
                # save renyi, avg seq len, vocab size to a csv
                with open(os.path.join(out_path, "eval_results.csv"), "w") as f:
                    f.write(f"renyi_eff_2.5,renyi_sim_eff_2.5,renyi_eff_3.0,avg_seq_len,vocab_size\n")
                    f.write(f"{renyi_25},{sim_renyi_25},{renyi_30},{seq_len}\n")

    import pandas as pd
    result_df = pd.DataFrame(result_dict)
    if output_path:
        result_df.to_csv(os.path.join(output_path, "intrinsic_tokenizer_eval_results.csv"), sep="\t")
    else:
        result_df.to_csv("eval_results.csv", sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', default=None, type=str)
    args = parser.parse_args()
    main(output_path=args.output_path)
