"""
KorNLI Training - 학습 루프
"""

import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from config import DEVICE
from evaluate import evaluate


def train_epoch(model: PreTrainedModel, dataloader: DataLoader, optimizer, scheduler=None, device=None) -> float:
    """한 에폭 학습 - 평균 손실 반환"""
    if device is None:
        device = DEVICE

    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    return total_loss / len(dataloader)


def train(
    model: PreTrainedModel,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer,
    epochs: int,
    scheduler=None,
    early_stopping_patience: int = None,
    checkpoint_dir: str = None,
    device=None,
) -> dict:
    """전체 학습 - 결과 dict 반환"""
    if device is None:
        device = DEVICE

    best_accuracy = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        dev_accuracy, _, _ = evaluate(model, dev_loader, device)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f}")

        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_epoch = epoch + 1
            patience_counter = 0

            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pt")
        else:
            patience_counter += 1

        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return {"best_accuracy": best_accuracy, "best_epoch": best_epoch}
