from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    num_epochs: int,
) -> None:
    device = model.device
    num_training_steps = len(train_dataloader) * num_epochs
    progress_bar = tqdm(range(num_training_steps))
    print("Setting model to train mode...")
    model.train()
    print("Starting training...")
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids, attention_mask, y = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["y"],
            )
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_description(f"Epoch {epoch+1}")
            progress_bar.set_postfix({"train_loss": loss.item()})

            progress_bar.update(1)


def evaluate(
    model: nn.Module, eval_dataloader: DataLoader, num_classes: int
) -> torch.Tensor:
    device = model.device
    num_validation_steps = len(eval_dataloader)
    progress_bar = tqdm(range(num_validation_steps))
    print("Setting model to eval mode...")
    model.eval()
    print("Starting evaluation...")
    eval_softmax_outputs = torch.empty(
        len(eval_dataloader), num_classes, dtype=torch.float64
    ).to(device)
    index = 0
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids, attention_mask, y = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["y"],
        )
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = outputs.loss

        logits = outputs.logits
        softmax_outputs = F.softmax(logits, dim=1)
        offset = softmax_outputs.size(0)
        eval_softmax_outputs[index : index + offset] = softmax_outputs
        index += offset

        progress_bar.set_postfix({"val_loss": loss.item()})
        progress_bar.update(1)

    print("Setting model back to train mode...")
    model.train()

    # use eval_softmax_outputs to calculate metrics
    return eval_softmax_outputs
