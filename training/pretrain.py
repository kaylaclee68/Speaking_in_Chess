import transformers
import argparse
from datasets import load_dataset

def main(args):
    dataset = load_dataset(args.dataset, streaming=args.stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    main(args)

