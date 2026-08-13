"""
KorNLI Dataset - 데이터 로딩 및 Dataset 클래스
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

from preprocess import preprocess_text
from config import LABEL2ID


def load_data(file_path: str, preprocess: bool = False) -> pd.DataFrame:
    """TSV 데이터 파일 로드"""
    df = pd.read_csv(file_path, sep="\t", quoting=3, on_bad_lines="skip")
    # 결측값 제거
    df = df.dropna(subset=["sentence1", "sentence2"])
    if preprocess:
        df["sentence1"] = df["sentence1"].apply(preprocess_text)
        df["sentence2"] = df["sentence2"].apply(preprocess_text)
    return df


class NLIDataset(Dataset):
    """NLI 태스크용 PyTorch Dataset"""

    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_length: int, label2id: dict = None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["sentence1"],
            row["sentence2"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.label2id and "gold_label" in row:
            item["labels"] = torch.tensor(self.label2id[row["gold_label"]])
        return item


def create_dataloaders(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    batch_size: int,
    test_df: pd.DataFrame = None,
) -> tuple:
    """DataLoader 일괄 생성"""
    train_dataset = NLIDataset(train_df, tokenizer, max_length, LABEL2ID)
    dev_dataset = NLIDataset(dev_df, tokenizer, max_length, LABEL2ID)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    if test_df is not None:
        test_dataset = NLIDataset(test_df, tokenizer, max_length, LABEL2ID)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, dev_loader, test_loader

    return train_loader, dev_loader
