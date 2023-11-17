import transformers
import argparse
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Config
from tokenizers import Tokenizer
from torch.utils.data import Dataset

def main(args):
    dataset = load_dataset(args.dataset, streaming=args.stream).shuffle(seed=args.seed)

    split_dataset = dataset["train"].train_test_split(test_size=0.00025, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    train_dataset = Dataset(split_dataset['train'].with_format('torch'))

    val_dataset = Dataset(split_dataset['val'].with_format('torch'))

    tokenizer = Tokenizer.from_file(args.tokenizer)

    training_arguments = TrainingArguments(
        output_dir=args.outdir,
        do_train=True,
        evaluation_strategy='step',
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        num_train_epochs=args.epochs,
        logging_first_step=True,
        logging_steps=500,
        save_strategy='steps',
        save_steps=1000,
        seed=args.seed,
        data_seed=args.dataseed,
        bf16=args.bf16,
        group_by_length=args.group_by_length
    )

    gpt2_config = GPT2Config(
        vocab_size=23296,
        n_positions=768,
        bos_token_id=Tokenizer.token_to_id("<eos>"),
        eos_token_id=Tokenizer.token_to_id("<eos>")
    )

    model = GPT2LMHeadModel(gpt2_config)

    trainer = Trainer(model=model, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset, args=training_arguments)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument("--dataseed", type=int, required=False, default=123)
    parser.add_argument("--train-batch-size", type=int, required=True, default=96)
    parser.add_argument("--eval-batch-size", type=int, required=True, default=16)
    parser.add_argument("--gradient-accumulation-steps", "-g", type=int, required=True, default=5)
    parser.add_argument("--learning-rate", "-l", type=float, required=True, default=2e-5)
    parser.add_argument("--weight-decay", "-w", type=float, required=True, default=0.0)
    parser.add_argument("--group-by-length", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--adam-beta1", type=float, required=False, default=0.9)
    parser.add_argument("--adam-beta2", type=float, required=False, default=0.95)
    parser.add_argument("--epochs", type=int, required=False, default=1)
    
    


    args = parser.parse_args()

    main(args)

