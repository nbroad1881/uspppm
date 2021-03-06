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
from datasets import Dataset, disable_progress_bar

from cocolm.tokenization_cocolm import COCOLMTokenizer


def get_folds(df, k_folds=5, stratify_on="score", groups="anchor"):

    if stratify_on and groups:
        sgkf = StratifiedGroupKFold(n_splits=k_folds)
        return [
            val_idx
            for _, val_idx in sgkf.split(
                df, y=df[stratify_on].astype(str), groups=df[groups]
            )
        ]
    elif groups:
        gkf = GroupKFold(n_splits=k_folds)
        return [val_idx for _, val_idx in gkf.split(df, groups=df[groups])]
    elif stratify_on:
        skf = StratifiedKFold(n_splits=k_folds)
        return [val_idx for _, val_idx in skf.split(df, y=df[stratify_on])]
    kf = KFold(n_splits=k_folds)
    return [val_idx for _, val_idx in kf.split(df)]


@dataclass
class DataModule:

    cfg: dict = None

    def __post_init__(self):
        if self.cfg is None:
            raise ValueError

        self.data_dir = Path(self.cfg["data_dir"])

        train_df = pd.read_csv(self.data_dir / self.cfg["train_file"])
        train_df["section"] = train_df["context"].apply(lambda x: f"[{x[0]}]")
        
        if self.cfg["prompt"] == "detailed":
            details = pd.read_csv(self.data_dir/"detailed-context.csv")
            train_df = train_df.merge(details, on="context")
            train_df["context"] = train_df["details"]
        else:
            train_df["context"] = train_df["title"]
        

        if self.cfg["ignore_data"]:
            ignore = pd.read_csv(self.data_dir / "ignore.csv")
            train_df = train_df[~train_df.id.isin(ignore.id)]

        self.train_df = train_df.sample(frac=1, random_state=42, ignore_index=True)
        if self.cfg["DEBUG"]:
            self.train_df = self.train_df.sample(n=1000, ignore_index=True)

        if "cocolm" in self.cfg["model_name_or_path"]:
            tok_class = COCOLMTokenizer
        else:
            tok_class = AutoTokenizer
        
        self.tokenizer = tok_class.from_pretrained(
            self.cfg["model_name_or_path"],
            use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        )

        if "fold" not in self.train_df:
            self.fold_idxs = get_folds(
                self.train_df,
                k_folds=self.cfg["k_folds"],
                stratify_on=self.cfg["stratify_on"],
                groups=self.cfg["fold_groups"],
            )
        else:
            self.fold_idxs = [
                self.train_df[self.train_df.fold == f].index.tolist()
                for f in range(self.train_df.fold.max()+1)
            ]

        if self.cfg["prompt"] == "section":
            sections = ["B", "H", "G", "C", "A", "F", "E", "D" ]
            new_tokens = [f"[{x}]" for x in sections]
            self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    def prepare_datasets(self, add_idx=False):

        if self.cfg["train_file"]:
            pass
        elif self.cfg["detailed_cpc"]:
            cpc_texts = get_cpc_details(self.data_dir)
            self.train_df["context"] = self.train_df["context"].map(cpc_texts)
        else:
            cpc_categories = get_cpc_categories()
            self.train_df["context"] = self.train_df["context"].apply(
                lambda x: cpc_categories[x[0]]
            )

        id_col = {}
        if add_idx:
            self.train_df["idx"] = range(len(self.train_df))
            self.train_df[["idx", "id"]].to_csv("id2idx.csv", index=False)
            id_col = {"idx": self.train_df["idx"]}

        self.raw_ds = Dataset.from_dict(
            {
                "anchor": self.train_df.anchor,
                "target": self.train_df.target,
                "label": self.train_df.score,
                "context": self.train_df.context,
                "fold": self.train_df.fold,
                "section": self.train_df.section,
                **id_col,
            }
        )

        remove_columns = self.raw_ds.column_names

        if add_idx:
            remove_columns.remove("idx")

        disable_progress_bar()

        self.dataset = self.raw_ds.map(
            self.tokenize,
            batched=False,
            remove_columns=remove_columns,
            num_proc=self.cfg["num_proc"],

        )

    def get_train_dataset(self, fold):
        idxs = list(chain(*[i for f, i in enumerate(self.fold_idxs) if f != fold]))
        return self.dataset.select(idxs)

    def get_eval_dataset(self, fold):
        idxs = self.fold_idxs[fold]
        print("Unique eval fold values:", self.raw_ds.select(idxs).unique("fold"))
        return self.dataset.select(idxs)


    def tokenize(self, example):

        if not self.cfg["prompt"] == "text-to-text":
            sep = self.tokenizer.sep_token

        ctx = example["context"]
        if self.cfg["lowercase"]:
            ctx = ctx.lower()

        if self.cfg["prompt"] == "natural":
            prompt = [
                f"How similar is '{example['anchor']}' compared to '{example['target']}' given the context '{ctx}'?"
            ]

        elif self.cfg["prompt"] == "token_type" or self.cfg["prompt"] == "detailed":
            prompt = [
                example["anchor"],
                example["target"] + sep + ctx,
            ]
        elif self.cfg["prompt"] == "spaces":
            prompt = [" ".join([example["anchor"], example["target"], ctx])]
        elif self.cfg["prompt"] == "sep":
            prompt = (
                example["anchor"] + sep + example["target"] + sep + ctx
            )
            prompt = [prompt]
        elif self.cfg["prompt"] == "section":
            sep = example["section"]
            prompt = (
                example["anchor"] + sep + example["target"] + sep + ctx
            )
            prompt = [prompt]
        elif self.cfg["prompt"] == "text-to-text":
            prompt = [
                f"compare: '{example['anchor']}' and '{example['target']}' given the context '{ctx}'. There is <extra_id_0> similarity."
            ]
            mapping = {
                100: "???maximum",
                75: "???very",
                50: "???some",
                25: "???little",
                0: "???no"
            }
            example["label"] = self.tokenizer("<extra_id_0>"+mapping[int(example["label"]*100)]).input_ids

        if self.cfg["no-semicolon"]:
            prompt = [p.replace(";", "") for p in prompt]
        if "cocolm" in self.cfg["model_name_or_path"]:
            tokenized = self.tokenizer.encode_plus(*prompt)
            tokenized["attention_mask"] = [0] + [1]*(len(tokenized["input_ids"])-2) + [0]
        else:
            tokenized = self.tokenizer(*prompt, padding=False, truncation="only_second", max_length=self.cfg["max_seq_length"])
        tokenized["label"] = example["label"]
        return tokenized


def get_cpc_details(data_dir):
    contexts = []
    pattern = "[A-Z]\d+"
    for file_name in os.listdir(data_dir / "CPCSchemeXML202105"):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(list(chain(*contexts))))
    results = {}
    for cpc in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
        with open(data_dir / f"CPCTitleList202202/cpc-section-{cpc}_20220201.txt") as f:
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
