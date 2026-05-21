import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from madewithml.config import STOPWORDS


class TextDataset(Dataset):
    """Dataset for tokenized text samples."""

    def __init__(self, ids: List[np.ndarray], masks: List[np.ndarray], targets: np.ndarray):
        self.ids = [np.asarray(item, dtype=np.int64) for item in ids]
        self.masks = [np.asarray(item, dtype=np.int64) for item in masks]
        self.targets = np.asarray(targets, dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "ids": self.ids[idx],
            "masks": self.masks[idx],
            "targets": self.targets[idx],
        }


def load_data(dataset_loc: str, num_samples: int = None) -> pd.DataFrame:
    """Load data from source into a pandas DataFrame."""
    df = pd.read_csv(dataset_loc)
    if num_samples is not None:
        df = df.sample(n=num_samples, random_state=1234).reset_index(drop=True)
    return df


def stratify_split(
    df: pd.DataFrame,
    stratify: str,
    test_size: float,
    shuffle: bool = True,
    seed: int = 1234,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into train and test splits with stratification."""

    def _add_split(frame: pd.DataFrame) -> pd.DataFrame:
        train, test = train_test_split(frame, test_size=test_size, shuffle=shuffle, random_state=seed)
        train["_split"] = "train"
        test["_split"] = "test"
        return pd.concat([train, test])

    def _filter_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
        return frame[frame["_split"] == split].drop("_split", axis=1)

    grouped = df.groupby(stratify).apply(_add_split)
    train_df = grouped.groupby(stratify).apply(_filter_split, split="train").reset_index(drop=True)
    test_df = grouped.groupby(stratify).apply(_filter_split, split="test").reset_index(drop=True)

    if shuffle:
        train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, test_df


def clean_text(text: str, stopwords: List = STOPWORDS) -> str:
    """Clean raw text string."""
    text = text.lower()
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub(" ", text)
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text)
    text = re.sub(" +", " ", text)
    text = text.strip()
    text = re.sub(r"http\S+", "", text)
    return text


def tokenize(texts: List[str], vocab: Dict[str, int]) -> Dict[str, List[np.ndarray]]:
    """Tokenize text strings using a simple vocabulary."""
    ids = []
    masks = []
    for text in texts:
        token_ids = [vocab.get(token, vocab.get("<unk>")) for token in text.split()]
        if len(token_ids) == 0:
            token_ids = [vocab.get("<unk>")]
        mask = [1] * len(token_ids)
        ids.append(np.array(token_ids, dtype=np.int64))
        masks.append(np.array(mask, dtype=np.int64))
    return {"ids": ids, "masks": masks}


def preprocess(df: pd.DataFrame, class_to_index: Dict, vocab: Dict[str, int]) -> Dict:
    """Preprocess the data in our dataframe."""
    frame = df.copy()
    frame["text"] = frame.title.fillna("") + " " + frame.description.fillna("")
    frame["text"] = frame.text.apply(clean_text)
    frame = frame.drop(columns=["id", "created_on", "title", "description"], errors="ignore")
    frame = frame[["text", "tag"]]
    frame["tag"] = frame["tag"].map(class_to_index)
    tokenized = tokenize(frame["text"].tolist(), vocab)
    return {
        "ids": tokenized["ids"],
        "masks": tokenized["masks"],
        "targets": np.array(frame["tag"].tolist(), dtype=np.int64),
    }


class CustomPreprocessor:
    """Custom preprocessor class."""

    def __init__(self, class_to_index: Dict = None, vocab: Dict[str, int] = None):
        self.class_to_index = class_to_index or {}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.vocab = vocab or {}

    def _build_vocab(self, texts: List[str], max_tokens: int = 10000) -> Dict[str, int]:
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        vocab = {"<pad>": 0, "<unk>": 1}
        for idx, (token, _) in enumerate(counter.most_common(max_tokens), start=2):
            vocab[token] = idx
        return vocab

    def fit(self, df: pd.DataFrame):
        frame = df.copy()
        frame["text"] = frame.title.fillna("") + " " + frame.description.fillna("")
        frame["text"] = frame.text.apply(clean_text)
        tags = sorted(frame["tag"].unique())
        self.class_to_index = {tag: i for i, tag in enumerate(tags)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.vocab = self._build_vocab(frame["text"].tolist())
        return self

    def transform(self, df: pd.DataFrame) -> TextDataset:
        outputs = preprocess(df, class_to_index=self.class_to_index, vocab=self.vocab)
        return TextDataset(ids=outputs["ids"], masks=outputs["masks"], targets=outputs["targets"])
