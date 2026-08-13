#!/usr/bin/env python3
"""
KorNLI Main - CLI 진입점

Usage:
    python main.py --preset default --data_dir ../data
"""

import argparse
import json

from config import get_config, DEVICE, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_CHECKPOINT_DIR
from dataset import load_data, create_dataloaders
from model import load_tokenizer, load_model, create_optimizer, create_scheduler, load_checkpoint
from train import train
from evaluate import evaluate, get_classification_report
from predict import predict, create_submission, save_submission


def main():
    parser = argparse.ArgumentParser(description="KorNLI Training Pipeline")
    parser.add_argument("--preset", type=str, default="default",
                        help="Config preset (default)")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    # Config
    overrides = {k: v for k, v in [("epochs", args.epochs), ("batch_size", args.batch_size), ("learning_rate", args.learning_rate)] if v}
    config = get_config(args.preset, **overrides)
    print(f"Config: {args.preset}, Device: {DEVICE}")

    # Data
    train_df = load_data(f"{args.data_dir}/{config['train_file']}", preprocess=config["preprocess"])
    dev_df = load_data(f"{args.data_dir}/{config['dev_file']}", preprocess=config["preprocess"])
    test_df = load_data(f"{args.data_dir}/{config['test_file']}", preprocess=config["preprocess"])
    print(f"Data: train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}")

    # Model
    tokenizer = load_tokenizer(config["model_name"])
    model = load_model(config["model_name"], classifier_dropout=config["classifier_dropout"])

    # DataLoaders
    train_loader, dev_loader, test_loader = create_dataloaders(
        train_df, dev_df, tokenizer, config["max_length"], config["batch_size"], test_df
    )

    # Train
    if not args.skip_train:
        optimizer = create_optimizer(model, config["learning_rate"], config["weight_decay"])
        num_steps = len(train_loader) * config["epochs"]
        scheduler = create_scheduler(optimizer, num_steps, config["warmup_ratio"])

        train(model, train_loader, dev_loader, optimizer, config["epochs"],
              scheduler=scheduler, early_stopping_patience=config["early_stopping_patience"],
              checkpoint_dir=args.checkpoint_dir)

        try:
            tokenizer.save_pretrained(args.checkpoint_dir)
        except TypeError:
            pass  # KoBertTokenizer doesn't support filename_prefix argument
        with open(f"{args.checkpoint_dir}/config.json", "w") as f:
            json.dump(config, f, indent=2)

    # Load best & Evaluate
    model = load_checkpoint(model, f"{args.checkpoint_dir}/best_model.pt")
    accuracy, preds, labels = evaluate(model, dev_loader)
    print(f"\nDev Accuracy: {accuracy:.4f}")
    print(get_classification_report(labels, preds))

    # Predict & Save
    test_preds = predict(model, test_loader)
    submission_df = create_submission(test_preds)
    save_submission(submission_df, f"{args.output_dir}/submission.csv")
    print(f"Submission saved: {args.output_dir}/submission.csv")


if __name__ == "__main__":
    main()
