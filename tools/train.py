import wandb
import datasets
import torch.utils.data
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from responsible_ai_audit.data.get_data import (
    SentimentDataset,
    get_train_val_split,
    filter_functions_mapping,
)
from responsible_ai_audit.metrics import accuracy_score
from torch.optim import AdamW
from argparse import ArgumentParser
from pathlib import Path
from transformers import get_linear_schedule_with_warmup
from transformers import set_seed


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
) -> None:
    device = model.device
    num_training_steps = wandb.config["max_steps"]
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    global_step = 1
    epoch = 1
    loss_fn = nn.CrossEntropyLoss()
    metrics = {
        "train_loss": 0.0,
        "val_loss": 0.0,
        "train_accuracy": 0.0,
        "val_accuracy": 0.0,
    }
    train_batches = 0
    val_batches = 0

    while True:
        for batch in train_dataloader:
            if global_step == num_training_steps:
                progress_bar.close()
                torch.save(
                    model.state_dict(),
                    wandb.config["model_save_dir"].joinpath(f"step_{global_step}.pth"),
                )
                return

            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids, attention_mask, y = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["y"],
            )

            wandb.config["optimizer"].zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs[-1]
                loss = loss_fn(logits, y)

            loss.backward()
            wandb.config["optimizer"].step()
            wandb.config["lr_scheduler"].step()

            softmax_outputs = F.softmax(logits, dim=1)
            y_pred = torch.argmax(softmax_outputs, dim=1)
            accuracy = accuracy_score(y, y_pred)
            metrics["train_loss"] += loss.item()
            metrics["train_accuracy"] += accuracy
            train_batches += 1

            if global_step > 0 and global_step % wandb.config["log_interval"] == 0:
                model.eval()
                with torch.no_grad():
                    for batch in eval_dataloader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        input_ids, attention_mask, y = (
                            batch["input_ids"],
                            batch["attention_mask"],
                            batch["y"],
                        )
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            outputs = model(
                                input_ids=input_ids, attention_mask=attention_mask
                            )
                            logits = outputs[-1]
                            val_loss = loss_fn(logits, y)

                        softmax_outputs = F.softmax(logits, dim=1)
                        y_pred = torch.argmax(softmax_outputs, dim=1)
                        val_accuracy = accuracy_score(y, y_pred)
                        metrics["val_loss"] += val_loss.item()
                        metrics["val_accuracy"] += val_accuracy
                        val_batches += 1

                model.train()

                wandb.config["model_save_dir"].mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    wandb.config["model_save_dir"].joinpath(f"step_{global_step}.pth"),
                )

                metrics["train_loss"] /= train_batches
                metrics["val_loss"] /= val_batches
                metrics["train_accuracy"] /= train_batches
                metrics["val_accuracy"] /= val_batches

                wandb.log(metrics, step=global_step)
                progress_bar.set_postfix(metrics)

                # reset metrics and batch counters
                metrics = {
                    "train_loss": 0.0,
                    "val_loss": 0.0,
                    "train_accuracy": 0.0,
                    "val_accuracy": 0.0,
                }
                train_batches = 0
                val_batches = 0

            global_step += 1
            progress_bar.update(1)

        progress_bar.set_description(f"Epoch {epoch}")
        epoch += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--model_save_dir", type=str, default="./models")
    args = parser.parse_args()
    wandb.init(project=args.wandb_project)

    # capture a dictionary of hyperparameters with config
    wandb.config = {
        "lr": args.lr,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "log_interval": args.log_interval,
        "model_save_dir": Path(args.model_save_dir),
    }
    set_seed(wandb.config["seed"])

    torch.set_float32_matmul_precision("high")

    dataset = datasets.load_dataset("social_bias_frames")
    num_labels = 3
    model = "distilbert/distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model)
    config = AutoConfig.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(
        model, device_map="cuda", num_labels=num_labels
    )

    train_df, val_df = get_train_val_split(
        dataset,
        filter_functions_mapping[args.subset],
        random_state=wandb.config["seed"],
    )
    train_dataset = SentimentDataset(train_df, tokenizer)
    val_dataset = SentimentDataset(val_df, tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=wandb.config["batch_size"],
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=wandb.config["batch_size"]
    )

    wandb.config.update({"optimizer": AdamW(model.parameters(), lr=wandb.config["lr"])})
    wandb.config.update(
        {
            "lr_scheduler": get_linear_schedule_with_warmup(
                wandb.config["optimizer"],
                num_warmup_steps=0,
                num_training_steps=wandb.config["max_steps"],
            )
        }
    )

    train(model, train_loader, val_loader)
