import os
import json
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np


class MOCHADataset(Dataset):
    """
    Minimal MOCHA wrapper for the standard pipeline (no concept supervision).
    - Inputs: concatenation of context/question/reference/candidate
    - Labels: discretized bins derived from continuous 'score'
    """
    # Shared bin edges across splits to ensure consistent mapping
    _cached_edges: Optional[np.ndarray] = None

    def __init__(self, split: str, tokenizer, max_len: int, bins: int = 5, base_dir: Optional[str] = None):
        assert split in ("train", "val", "test"), f"Unsupported split: {split}"
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bins = bins

        # Resolve repo root and expected dataset locations
        SELF_DIR = os.path.dirname(os.path.abspath(__file__))
        MAIN_DIR = os.path.dirname(SELF_DIR)
        ROOT_DIR = os.path.dirname(MAIN_DIR)
        DEFAULT_DIR = os.path.join(ROOT_DIR, "dataset", "mocha")
        self.data_dir = base_dir or DEFAULT_DIR

        # Candidate JSON paths (support common tar layout variants)
        candidates = [
            # flat under dataset/mocha
            ("dev.json", "test.json"),
            # nested as dataset/mocha/data/mocha
            (os.path.join("data", "mocha", "dev.json"), os.path.join("data", "mocha", "test.json")),
        ]

        dev_path = test_path = None
        for dev_rel, test_rel in candidates:
            d = os.path.join(self.data_dir, dev_rel)
            t = os.path.join(self.data_dir, test_rel)
            if os.path.exists(d) and os.path.exists(t):
                dev_path, test_path = d, t
                break
        if dev_path is None or test_path is None:
            raise FileNotFoundError(f"MOCHA files not found under {self.data_dir}. "
                                    f"Tried {candidates}")

        # Load JSON
        with open(dev_path, "r") as f:
            dev_items: List[Dict[str, Any]] = json.load(f)
        with open(test_path, "r") as f:
            test_items: List[Dict[str, Any]] = json.load(f)

        # Split dev into train/val
        cut = int(0.8 * len(dev_items))
        if split == "train":
            self.items = dev_items[:cut]
        elif split == "val":
            self.items = dev_items[cut:]
        else:
            self.items = test_items

        # Precompute or load bin edges using full dev set to ensure consistency
        if MOCHADataset._cached_edges is None:
            all_scores = np.array([self._extract_score(x) for x in dev_items], dtype=float)
            # Quantile-based bins to roughly balance classes
            quantiles = np.linspace(0.0, 1.0, self.bins + 1)
            edges = np.quantile(all_scores, quantiles)
            # Ensure strict monotonicity to avoid identical edges
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + 1e-6
            MOCHADataset._cached_edges = edges
        self._edges = MOCHADataset._cached_edges

        # Dataset metadata
        self.final_label = ['quality']
        self.final_label_vals = list(range(self.bins))  # 0..bins-1
        # No native concepts; provide a small latent width for the standard head
        self.concepts = [f"latent_{i}" for i in range(4)]
        self.concept_vals = [0, 1, 2]  # unused in standard pipeline

        # Keep all indices
        self.indices = list(range(len(self.items)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        i = self.indices[index]
        ex = self.items[i]
        text = self._compose_text(ex)
        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        label_idx = self._score_to_bin(self._extract_score(ex))
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "label": torch.tensor(label_idx, dtype=torch.long),
        }

    def _extract_score(self, ex: Dict[str, Any]) -> float:
        # MOCHA 'score' as continuous label
        val = ex.get("score", 0.0)
        try:
            return float(val)
        except Exception:
            return 0.0

    def _compose_text(self, ex: Dict[str, Any]) -> str:
        context = str(ex.get("context", "")).strip()
        question = str(ex.get("question", "")).strip()
        reference = str(ex.get("reference", "")).strip()
        candidate = str(ex.get("candidate", "")).strip()
        # Concatenate segments with clear separators
        parts = []
        if context:
            parts.append(f"[CTX] {context}")
        if question:
            parts.append(f"[Q] {question}")
        if reference:
            parts.append(f"[REF] {reference}")
        if candidate:
            parts.append(f"[CAND] {candidate}")
        return " [SEP] ".join(parts) if parts else ""

    def _score_to_bin(self, score: float) -> int:
        # Map continuous score to bin index using cached edges
        edges = self._edges
        # np.digitize returns indices in 1..len(edges)-1; subtract 1 to get 0-based
        idx = int(np.digitize([score], edges[1:-1], right=False)[0])
        # Clamp into 0..bins-1
        if idx < 0:
            idx = 0
        if idx >= self.bins:
            idx = self.bins - 1
        return idx


