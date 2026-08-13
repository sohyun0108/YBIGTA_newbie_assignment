"""
KorNLI Prediction - 추론 및 submission 생성
"""

import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from config import DEVICE, ID2LABEL


def predict(model: PreTrainedModel, dataloader: DataLoader, device: torch.device = None) -> list:
    """모델 예측 - 예측 ID 리스트 반환"""
    if device is None:
        device = DEVICE

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend(preds)

    return predictions


def create_submission(predictions: list, id2label: dict = None) -> pd.DataFrame:
    """submission DataFrame 생성"""
    if id2label is None:
        id2label = ID2LABEL
    pred_labels = [id2label[p] for p in predictions]
    return pd.DataFrame({"id": range(len(pred_labels)), "label": pred_labels})


def save_submission(submission_df: pd.DataFrame, output_path: str) -> None:
    """submission CSV 저장"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
