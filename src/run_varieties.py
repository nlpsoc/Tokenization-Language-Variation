import argparse
import subprocess

tasks = ["sadiri"]


def main(task, model_path, seed, output_dir):

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
            "--gradient_checkpointing", False,
            "--saving_step", "1000",
            "--mask", "0",
            "--seed", seed,
            "--corpus", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle",
            "--loss", "SupConLoss"
        ]
        result = subprocess.run(command, capture_output=True, text=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task', default='sadiri', type=str, choices=tasks)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)

    args = parser.parse_args()
    main(task=args.task, model_path=args.model_path, seed=args.seed, output_dir=args.output_dir)
