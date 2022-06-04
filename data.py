import os
import re
from pathlib import Path
from itertools import chain
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GroupKFold,
    StratifiedKFold,
    KFold,
)
from transformers import AutoTokenizer
from datasets import Dataset


def get_folds(df, kfolds=5, stratify_on="score", groups="anchor"):

    if stratify_on and groups:
        sgkf = StratifiedGroupKFold(n_splits=kfolds)
        return [
            val_idx
            for _, val_idx in sgkf.split(df, y=df[stratify_on], groups=df[groups])
        ]
    elif groups:
        gkf = GroupKFold(n_splits=kfolds)
        return [val_idx for _, val_idx in gkf.split(df, groups=df[groups])]
    elif stratify_on:
        skf = StratifiedKFold(n_splits=kfolds)
        return [val_idx for _, val_idx in skf.split(df, y=df[stratify_on])]
    kf = KFold(n_splits=kfolds)
    return [val_idx for _, val_idx in kf.split(df)]


@dataclass
class DataModule:

    cfg: dict = None

    def __post_init__(self):
        if self.cfg is None:
            raise ValueError

        self.data_dir = Path(self.cfg["data_dir"])

        train_df = pd.read_csv(self.data_dir / "train.csv")

        self.train_df = train_df.sample(frac=1, random_state=42)
        if self.cfg["DEBUG"]:
            self.train_df = self.train_df.sample(n=1000)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model_name_or_path"],
            use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        )

        self.fold_idxs = get_folds(
            self.train_df,
            kfolds=self.cfg["kfolds"],
            stratify_on=self.cfg["stratify_on"],
            groups=self.cfg["fold_groups"],
        )

    def prepare_datasets(self):

        if self.cfg["detailed_cpc"]:
            cpc_texts = get_cpc_details(self.data_dir)
            self.train_df["context_text"] = self.train_df["context"].map(cpc_texts)
        else:
            cpc_categories = get_cpc_categories()
            self.train_df["context_text"] = self.train_df["context"].apply(
                lambda x: cpc_categories[x[0]]
            )

        raw_ds = Dataset.from_dict(
            {
                "anchor": self.train_df.anchor,
                "target": self.train_df.target,
                "label": self.train_df.score,
                "context": self.train_df.context,
            }
        )

        self.dataset = raw_ds.map(
            self.tokenize,
            batched=False,
            remove_columns=raw_ds.column_names,
            num_proc=self.cfg["num_proc"],
        )

    def get_train_dataset(self, fold):
        idxs = list(chain(*[i for f, i in enumerate(self.fold_idxs) if f != fold]))
        return self.dataset.select(idxs)

    def get_eval_dataset(self, fold):
        return self.dataset.select(self.fold_idxs[fold])

    def tokenize(self, example):

        if self.cfg["natural_language_prompt"]:
            prompt = f"How similar is '{example['anchor']}' compared to '{example['target']}' given the context '{example['context']}'"

        else:
            prompt = (
                example["anchor"]
                + self.tokenizer.sep_token
                + example["target"]
                + self.tokenizer.sep_token
                + example["context"]
            )

        if self.cfg["lowercase"]:
            prompt = prompt.lower()

        tokenized = self.tokenizer(prompt, padding=False)
        tokenized["label"] = example["label"]
        return tokenized


def get_cpc_details(data_dir):
    contexts = []
    pattern = "[A-Z]\d+"
    for file_name in os.listdir(data_dir / "cpc-data/CPCSchemeXML202105"):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(list(chain(*contexts))))
    results = {}
    for cpc in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
        with open(
            data_dir / f"cpc-data/CPCTitleList202202/cpc-section-{cpc}_20220201.txt"
        ) as f:
            s = f.read()
        pattern = f"{cpc}\t\t.+"
        result = re.findall(pattern, s)
        pattern = "^" + pattern[:-2]
        cpc_result = re.sub(pattern, "", result[0])
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f"{context}\t\t.+"
            result = re.findall(pattern, s)
            pattern = "^" + pattern[:-2]
            results[context] = cpc_result + ". " + re.sub(pattern, "", result[0])
    return results


def get_cpc_categories():
    return {
        "A": "Human Necessities",
        "B": "Operations and Transport",
        "C": "Chemistry and Metallurgy",
        "D": "Textiles",
        "E": "Fixed Constructions",
        "F": "Mechanical Engineering",
        "G": "Physics",
        "H": "Electricity",
        "Y": "Emerging Cross-Sectional Technologies",
    }