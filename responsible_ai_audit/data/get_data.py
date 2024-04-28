import datasets
import torch.utils.data
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from typing import Tuple, Callable, Dict, Any

from responsible_ai_audit.data.preprocessing import preprocess


def get_train_val_split(
    dataset: datasets.Dataset,
    filter_function: Callable[[pd.DataFrame], pd.Series],
    test_size: float = 0.2,
    random_state: int = 88,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = dataset["train"]
    validation = dataset["validation"]
    train_df = train.to_pandas()
    val_df = validation.to_pandas()
    # merge datasets and group by post to get majority label
    merged = pd.concat([train_df, val_df])
    mask = filter_function(merged)
    grouped = (
        merged[mask]
        .groupby("post")["offensiveYN"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )
    # split into 80/20 train/val split
    train_split, val_split = train_test_split(
        grouped, test_size=test_size, random_state=random_state
    )
    return train_split, val_split


def get_test_split(dataset: datasets.Dataset) -> pd.DataFrame:
    test = dataset["test"]
    test_df = test.to_pandas()
    grouped = (
        test_df.groupby("post")["offensiveYN"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )
    return grouped


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer) -> None:
        self.dataframe = dataframe
        self.dataframe = self.preprocess_data(self.dataframe)
        self.tokenizer = tokenizer
        self.id2label = {0: "no", 1: "neutral", 2: "yes"}  # offensiveYN
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.__post_init__()

    def __post_init__(self) -> None:
        self.batch_encoding = self.tokenizer(
            self.dataframe["post"].tolist(), return_tensors="pt", padding=True
        )

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[df["post"] != ""]  # remove empty posts
        df["post"] = df["post"].map(preprocess)
        id2class = {0: 0, 0.5: 1, 1: 2}
        df["offensiveYN"] = df["offensiveYN"].map(lambda x: id2class[float(x)])
        return df

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_ids": self.batch_encoding["input_ids"][idx],
            "attention_mask": self.batch_encoding["attention_mask"][idx],
            "y": self.dataframe["offensiveYN"].iat[idx],
        }


# other populations to explore - hisp man: 3k - liberal: 900, cons-1.8k, hisp woman: 4k - mod-liberal: 1k, other: 2k


# white male conservative - train/val = 5184/1296
def getWhiteMaleConsData(df: pd.DataFrame) -> pd.Series:
    return (
        (df["annotatorRace"] == "white")
        & (df["annotatorGender"] == "man")
        & (df["annotatorPolitics"] == "cons")
    )


# white male liberal - train/val = 10,404/2601
def getWhiteMaleLibData(df: pd.DataFrame) -> pd.Series:
    return (
        (df["annotatorRace"] == "white")
        & (df["annotatorGender"] == "man")
        & (df["annotatorPolitics"] == "liberal")
    )


# white female liberal - train/val = 14,557/3640
def getWhiteFemaleLibData(df: pd.DataFrame) -> pd.Series:
    return (
        (df["annotatorRace"] == "white")
        & (df["annotatorGender"] == "woman")
        & (df["annotatorPolitics"] == "liberal")
    )


# white female conservative - train/val = 756/190
def getWhiteFemaleConsData(df: pd.DataFrame) -> pd.Series:
    return (
        (df["annotatorRace"] == "white")
        & (df["annotatorGender"] == "woman")
        & (df["annotatorPolitics"] == "cons")
    )


# black female mod-liberal - train/val = 3025/757
def getBlackFemaleModlibData(df: pd.DataFrame) -> pd.Series:
    return (
        (df["annotatorRace"] == "black")
        & (df["annotatorGender"] == "woman")
        & (df["annotatorPolitics"] == "mod-liberal")
    )


if __name__ == "__main__":
    ###
    # sample use:
    dataset = datasets.load_dataset("social_bias_frames")
    model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model)
    config = AutoConfig.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model)
    train_df, val_df = get_train_val_split(dataset, getWhiteMaleConsData)
    train_dataset = SentimentDataset(train_df, tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    print(train_loader.dataset[0]["input_ids"])
    print(train_loader.dataset[0]["attention_mask"])
    ###
