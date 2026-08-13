"""
KorNLI Model - 모델 및 토크나이저 로딩
"""

import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from config import NUM_LABELS, DEVICE


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """토크나이저 로드"""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def load_model(model_name: str, classifier_dropout: float = None, device: torch.device = None) -> PreTrainedModel:
    """분류 모델 로드"""
    if device is None:
        device = DEVICE

    model_config = AutoConfig.from_pretrained(model_name, num_labels=NUM_LABELS, trust_remote_code=True)

    if classifier_dropout is not None:
        if hasattr(model_config, "classifier_dropout"):
            model_config.classifier_dropout = classifier_dropout
        elif hasattr(model_config, "hidden_dropout_prob"):
            model_config.hidden_dropout_prob = classifier_dropout

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config, trust_remote_code=True)
    model.to(device)
    return model


def create_optimizer(model: PreTrainedModel, learning_rate: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """AdamW 옵티마이저 생성"""
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def create_scheduler(optimizer: torch.optim.Optimizer, num_training_steps: int, warmup_ratio: float = 0.0):
    """Linear warmup 스케줄러 생성 (warmup_ratio > 0일 때만)"""
    if warmup_ratio <= 0:
        return None
    warmup_steps = int(num_training_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)


def save_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, save_dir: str) -> None:
    """모델과 토크나이저 저장"""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
    tokenizer.save_pretrained(save_dir)


def load_checkpoint(model: PreTrainedModel, checkpoint_path: str, device: torch.device = None) -> PreTrainedModel:
    """체크포인트에서 모델 가중치 로드"""
    if device is None:
        device = DEVICE
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model
