import os
import argparse
import subprocess
from styletokenizer.utility.custom_logger import log_and_flush

tasks = ["sadiri", "stel"]


def main(task, model_path, seed, output_dir):
    # print all set vars
    log_and_flush(f"task: {task}")
    log_and_flush(f"model_path: {model_path}")
    log_and_flush(f"seed: {seed}")
    log_and_flush(f"output_dir: {output_dir}")

    if task == "sadiri":
        command = [
            "python", "sadiri_main.py",
            "--train",
            "--validate",
            "--out_dir", output_dir,
            "--pretrained_model", model_path,
            "--train_data", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/train",
            "--dev_data", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/validation",
            "--learning_rate", "0.00001",
            "--batch_size", "128",
            "--epochs", "5",
            "--max_length", "512",
            "--grad_acc", "1",
            "--gradient_checkpointing", "False",
            "--saving_step", "100",
            "--mask", "0",
            "--seed", str(seed),
            "--corpus", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle",
            "--loss", "SupConLoss"
        ]
        result = subprocess.run(command)

    if task == "stel":
        from styletokenizer.utility.env_variables import set_cache
        set_cache()
        import sys
        # add STEL folder to path
        sys.path.append('../../STEL/src/')
        import torch
        from STEL import STEL
        from STEL.similarity import Similarity, cosine_sim
        from sentence_transformers import SentenceTransformer

        class SBERTSimilarity(Similarity):
            def __init__(self):
                super().__init__()
                self.model = SentenceTransformer(output_dir)
                self.model.to("cuda" if torch.cuda.is_available() else "cpu")

            def similarities(self, sentences_1, sentences_2):
                with torch.no_grad():
                    sentence_emb_1 = self.model.encode(sentences_1, show_progress_bar=False)
                    sentence_emb_2 = self.model.encode(sentences_2, show_progress_bar=False)
                return [cosine_sim(sentence_emb_1[i], sentence_emb_2[i]) for i in range(len(sentences_1))]

        STEL.eval_on_STEL(style_objects=[SBERTSimilarity()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task', default='sadiri', type=str, choices=tasks)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)

    args = parser.parse_args()
    main(task=args.task, model_path=args.model_path, seed=args.seed, output_dir=args.output_dir)
