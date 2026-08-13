"""
KorNLI Evaluation - 모델 평가
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from transformers import PreTrainedModel

from config import DEVICE, LABEL2ID


def evaluate(model: PreTrainedModel, dataloader: DataLoader, device: torch.device = None) -> tuple:
    """모델 평가 - (accuracy, predictions, true_labels) 반환"""
    if device is None:
        device = DEVICE

    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels.numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy, predictions, true_labels


def get_classification_report(true_labels: list, predictions: list) -> str:
    """Classification report 문자열 반환"""
    return classification_report(true_labels, predictions, target_names=list(LABEL2ID.keys()))
