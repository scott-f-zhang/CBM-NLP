import pandas as pd
import torch
from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    def __init__(self, split: str, tokenizer, max_len: int, variant: str = "manual", expand_concepts=None):
        # Internal loading aligned with CEBaBDataset, variants: manual / aug_manual / gen / aug_gen
        if expand_concepts is None:
            expand_concepts = (variant != "manual")

        if variant == "manual":
            frames = {
                "train": pd.read_csv("../dataset/imdb/IMDB-train-manual.csv"),
                "val": pd.read_csv("../dataset/imdb/IMDB-dev-manual.csv"),
                "test": pd.read_csv("../dataset/imdb/IMDB-test-manual.csv"),
            }
        elif variant == "gen":
            frames = {
                "train": pd.read_csv("../dataset/imdb/IMDB-train-generated.csv"),
                "val": pd.read_csv("../dataset/imdb/IMDB-dev-generated.csv"),
                "test": pd.read_csv("../dataset/imdb/IMDB-test-generated.csv"),
            }
        elif variant == "aug_manual":
            # Currently mirrors manual files but enables extra noisy concepts
            frames = {
                "train": pd.read_csv("../dataset/imdb/IMDB-train-manual.csv"),
                "val": pd.read_csv("../dataset/imdb/IMDB-dev-manual.csv"),
                "test": pd.read_csv("../dataset/imdb/IMDB-test-manual.csv"),
            }
        elif variant == "aug_gen":
            train_manual = pd.read_csv("../dataset/imdb/IMDB-train-manual.csv")
            val_manual = pd.read_csv("../dataset/imdb/IMDB-dev-manual.csv")
            test_manual = pd.read_csv("../dataset/imdb/IMDB-test-manual.csv")
            train_gen = pd.read_csv("../dataset/imdb/IMDB-train-generated.csv")
            val_gen = pd.read_csv("../dataset/imdb/IMDB-dev-generated.csv")
            test_gen = pd.read_csv("../dataset/imdb/IMDB-test-generated.csv")
            frames = {
                "train": pd.concat([train_manual, train_gen], ignore_index=True),
                "val": pd.concat([val_manual, val_gen], ignore_index=True),
                "test": pd.concat([test_manual, test_gen], ignore_index=True),
            }
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        df = frames[split]

        # Text and label columns per run_imdb
        self.data = df
        self.text = df["review"].astype(str)
        self.labels = df["sentiment"]

        # Gold concepts (4)
        self.acting_aspect = df["acting"]
        self.storyline_aspect = df["storyline"]
        self.emotional_aspect = df["emotional arousal"]
        self.cinematography_aspect = df["cinematography"]

        # Optional noisy concepts (4)
        self.extra = None
        if expand_concepts and all(col in df for col in [
            "soundtrack", "directing", "background setting", "editing"
        ]):
            self.extra = {
                "soundtrack": df["soundtrack"],
                "directing": df["directing"],
                "background": df["background setting"],
                "editing": df["editing"],
            }

        self.map_dict = {"Negative": 0, "Negative ": 0, "Positive": 1, "unknown": 2, "Unkown": 2}
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.variant = variant
        self.expand_concepts = expand_concepts

        # Keep all rows; no 'no majority' filtering used here
        self.indices = list(range(len(self.labels)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        i = self.indices[index]
        text = self.text[i]
        label = 1 if str(self.labels[i]).strip() == "Positive" else 0

        acting_concept = self.map_dict[str(self.acting_aspect[i]).strip()]
        storyline_concept = self.map_dict[str(self.storyline_aspect[i]).strip()]
        emotional_concept = self.map_dict[str(self.emotional_aspect[i]).strip()]
        cinematography_concept = self.map_dict[str(self.cinematography_aspect[i]).strip()]

        concept_labels = [acting_concept, storyline_concept, emotional_concept, cinematography_concept]

        if self.extra is not None:
            soundtrack_concept = self.map_dict[str(self.extra["soundtrack"][i]).strip()]
            directing_concept = self.map_dict[str(self.extra["directing"][i]).strip()]
            background_concept = self.map_dict[str(self.extra["background"][i]).strip()]
            editing_concept = self.map_dict[str(self.extra["editing"][i]).strip()]
            concept_labels = concept_labels + [
                soundtrack_concept, directing_concept, background_concept, editing_concept
            ]

        enc = self.tokenizer.encode_plus(
            str(text), add_special_tokens=True, max_length=self.max_len,
            truncation=True, padding="max_length", return_attention_mask=True, return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
            "acting_concept": torch.tensor(acting_concept, dtype=torch.long),
            "storyline_concept": torch.tensor(storyline_concept, dtype=torch.long),
            "emotional_concept": torch.tensor(emotional_concept, dtype=torch.long),
            "cinematography_concept": torch.tensor(cinematography_concept, dtype=torch.long),
            "concept_labels": torch.tensor(concept_labels, dtype=torch.long),
        }
        if self.extra is not None:
            item.update({
                "soundtrack_concept": torch.tensor(concept_labels[4], dtype=torch.long),
                "directing_concept": torch.tensor(concept_labels[5], dtype=torch.long),
                "background_concept": torch.tensor(concept_labels[6], dtype=torch.long),
                "editing_concept": torch.tensor(concept_labels[7], dtype=torch.long),
            })
        return item


