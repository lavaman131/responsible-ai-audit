from torch.optim.lr_scheduler import LinearLR
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
    getWhiteMaleConsData,
)
from responsible_ai_audit.metrics import accuracy_score
from torch.optim import AdamW
from argparse import ArgumentParser
from loguru import logger
from transformers import get_linear_schedule_with_warmup


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
) -> None:
    device = model.device
    num_training_steps = wandb.config["max_steps"]
    num_validation_steps = len(eval_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    logger.info("Setting model to train mode...")
    model.train()
    logger.info("Starting training...")
    global_step = 0
    epoch = 1
    loss_fn = nn.CrossEntropyLoss()
    while True:
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids, attention_mask, y = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["y"],
            )

            wandb.config["optimizer"].zero_grad()

            with torch.autocast(device_type="cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs[-1]
                loss = loss_fn(logits, y)

            loss.backward()
            wandb.config["optimizer"].step()
            wandb.config["lr_scheduler"].step()

            if global_step % wandb.config["log_interval"] == 0:
                softmax_outputs = F.softmax(logits, dim=1)
                y_pred = torch.argmax(softmax_outputs, dim=1)
                accuracy = accuracy_score(y, y_pred)
                metrics = {"train_loss": loss.item(), "train_accuracy": accuracy}
                wandb.log(metrics)
                progress_bar.set_postfix(metrics)

            global_step += 1

            if global_step >= wandb.config["max_steps"]:
                break

            if global_step % wandb.config["val_interval"] == 0:
                val_progress_bar = tqdm(range(num_validation_steps))
                logger.info("Setting model to eval mode...")
                model.eval()
                logger.info("Starting evaluation...")
                global_step = 0
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
                        with torch.no_grad():
                            loss = loss_fn(logits, y)

                    if global_step % wandb.config["log_interval"] == 0:
                        softmax_outputs = F.softmax(logits, dim=1)
                        y_pred = torch.argmax(softmax_outputs, dim=1)
                        accuracy = accuracy_score(y, y_pred)
                        metrics = {"val_loss": loss.item(), "val_accuracy": accuracy}
                        wandb.log(metrics)
                        val_progress_bar.set_postfix(metrics)

                    val_progress_bar.update(1)
                    global_step += 1

                logger.info("Setting model back to train mode...")
                model.train()

            progress_bar.update(1)

        progress_bar.set_description(f"Epoch {epoch}")
        epoch += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=500)
    args = parser.parse_args()
    wandb.init(project=args.wandb_project)
    # capture a dictionary of hyperparameters with config
    wandb.config = {
        "lr": args.lr,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "log_interval": args.log_interval,
        "val_interval": args.val_interval,
    }

    dataset = datasets.load_dataset("social_bias_frames")
    num_labels = 3
    model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model)
    config = AutoConfig.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(
        model, device_map="cuda", num_labels=num_labels
    )
    train_df, val_df = get_train_val_split(dataset, getWhiteMaleConsData)
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
    # optional: track gradients
    wandb.watch(model)

    train(model, train_loader, val_loader)
