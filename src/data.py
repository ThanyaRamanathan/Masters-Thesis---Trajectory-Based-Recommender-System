import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .utils import build_history_prompt

PAD_TOKEN = 0
INTENT_FEATURE_NAMES = [
    "seq_len_norm",
    "unique_poi_ratio",
    "repeat_visit_ratio",
    "dominant_cat_ratio",
    "recent_repeat",
    "mean_time_gap_norm",
    "last_time_sin",
    "last_time_cos",
]


def time_to_bin(ts: datetime, bins: int) -> int:
    seconds = ts.hour * 3600 + ts.minute * 60 + ts.second
    return int(seconds / (24 * 3600 / bins))


def build_intent_features(
    poi_idxs: List[int],
    cat_idxs: List[int],
    time_idxs: List[int],
    time_bins: int,
    max_seq_len: int,
) -> List[float]:
    if not poi_idxs:
        return [0.0] * len(INTENT_FEATURE_NAMES)

    seq_len_norm = min(len(poi_idxs) / max(max_seq_len, 1), 1.0)
    unique_poi_ratio = len(set(poi_idxs)) / len(poi_idxs)
    repeat_visit_ratio = 1.0 - unique_poi_ratio
    dominant_cat_ratio = Counter(cat_idxs).most_common(1)[0][1] / len(cat_idxs)
    recent_repeat = 1.0 if len(poi_idxs) > 1 and poi_idxs[-1] in poi_idxs[:-1] else 0.0

    if len(time_idxs) > 1:
        gaps = [abs(curr - prev) for prev, curr in zip(time_idxs, time_idxs[1:])]
        mean_gap = sum(gaps) / len(gaps) / max(time_bins, 1)
    else:
        mean_gap = 0.0

    last_time = time_idxs[-1] / max(time_bins, 1)
    angle = 2 * math.pi * last_time
    return [
        float(seq_len_norm),
        float(unique_poi_ratio),
        float(repeat_visit_ratio),
        float(dominant_cat_ratio),
        float(recent_repeat),
        float(mean_gap),
        float(math.sin(angle)),
        float(math.cos(angle)),
    ]


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        max_seq_len: int = 32,
        time_bins: int = 48,
        min_seq_len: int = 2,
        poi_col: Optional[str] = None,
        user_col: Optional[str] = None,
        cat_col: Optional[str] = None,
        time_col: Optional[str] = None,
        trajectory_col: Optional[str] = None,
        existing_maps: Optional[Dict] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.csv_path = csv_path
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.time_bins = time_bins
        self.poi_col = poi_col
        self.user_col = user_col
        self.cat_col = cat_col
        self.time_col = time_col
        self.trajectory_col = trajectory_col
        self.split = split
        self.max_samples = max_samples
        self.data = pd.read_csv(csv_path)
        self._resolve_columns()

        if self.time_col not in self.data.columns:
            raise ValueError(f"Time column '{self.time_col}' not found in {csv_path}")

        self.data[self.time_col] = pd.to_datetime(self.data[self.time_col], errors="coerce")
        self.data = self.data.dropna(subset=[self.user_col, self.poi_col, self.cat_col, self.time_col])
        sort_cols = [self.trajectory_col, self.time_col] if self.trajectory_col else [self.user_col, self.time_col]
        self.data = self.data.sort_values(sort_cols)

        self.poi2idx = {}
        self.cat2idx = {}
        self.time2idx = {i: i for i in range(time_bins)}
        self.idx2poi = {}
        self.idx2cat = {}

        self._build_vocabs(existing_maps)
        if split == "train":
            self.flow_map = self._build_flow_map()
        elif existing_maps:
            self.flow_map = existing_maps.get("flow_map", {})
        else:
            self.flow_map = {}
        self.samples = self._build_samples(existing_maps)

    def _resolve_columns(self):
        def pick(current, candidates, label):
            if current is not None:
                if current not in self.data.columns:
                    raise ValueError(f"{label} column '{current}' not found in {self.csv_path}")
                return current
            for candidate in candidates:
                if candidate in self.data.columns:
                    return candidate
            raise ValueError(f"Could not infer {label} column for {self.csv_path}. Tried: {candidates}")

        self.user_col = pick(self.user_col, ["user_id", "UserId", "user"], "user")
        self.poi_col = pick(self.poi_col, ["poi_id", "POI_id", "node_name/poi_id"], "POI")
        self.cat_col = pick(self.cat_col, ["category", "poi_category", "POI_catname", "POI_catid", "POI_catid_code"], "category")
        self.time_col = pick(self.time_col, ["timestamp", "time", "local_time", "UTC_time"], "time")
        if self.trajectory_col is None:
            self.trajectory_col = "trajectory_id" if "trajectory_id" in self.data.columns else None
        elif self.trajectory_col not in self.data.columns:
            raise ValueError(f"Trajectory column '{self.trajectory_col}' not found in {self.csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        poi_seq = torch.tensor(sample["poi_seq"], dtype=torch.long)
        cat_seq = torch.tensor(sample["cat_seq"], dtype=torch.long)
        time_seq = torch.tensor(sample["time_seq"], dtype=torch.long)
        flow_feat = torch.tensor(sample["flow_feat"], dtype=torch.float)
        intent_feat = torch.tensor(sample["intent_feat"], dtype=torch.float)
        return {
            "poi_seq": poi_seq,
            "cat_seq": cat_seq,
            "time_seq": time_seq,
            "flow_feat": flow_feat,
            "intent_feat": intent_feat,
            "history_text": sample["history_text"],
            "target_poi": torch.tensor(sample["target_poi"], dtype=torch.long),
            "target_cat": torch.tensor(sample["target_cat"], dtype=torch.long),
            "target_time": torch.tensor(sample["target_time"], dtype=torch.long),
        }

    def _build_vocabs(self, existing_maps: Optional[Dict]):
        if existing_maps is not None:
            self.poi2idx = existing_maps["poi2idx"]
            self.cat2idx = existing_maps["cat2idx"]
            self.idx2poi = existing_maps["idx2poi"]
            self.idx2cat = existing_maps["idx2cat"]
            return

        poi_counter = Counter(self.data[self.poi_col].astype(str))
        cat_counter = Counter(self.data[self.cat_col].astype(str))

        self.poi2idx = {"PAD": PAD_TOKEN}
        self.cat2idx = {"PAD": PAD_TOKEN}
        idx = 1
        for poi in poi_counter:
            self.poi2idx[poi] = idx
            idx += 1
        self.idx2poi = {idx: poi for poi, idx in self.poi2idx.items()}

        idx = 1
        for cat in cat_counter:
            self.cat2idx[cat] = idx
            idx += 1
        self.idx2cat = {idx: cat for cat, idx in self.cat2idx.items()}

    def _build_samples(self, existing_maps: Optional[Dict]):
        samples = []
        group_col = self.trajectory_col or self.user_col
        groups = self.data.groupby(group_col)

        for _, group in groups:
            poi_list = group[self.poi_col].astype(str).tolist()
            cat_list = group[self.cat_col].astype(str).tolist()
            time_list = group[self.time_col].tolist()
            if len(poi_list) < self.min_seq_len + 1:
                continue

            poi_idxs = [self.poi2idx.get(p, PAD_TOKEN) for p in poi_list]
            cat_idxs = [self.cat2idx.get(c, PAD_TOKEN) for c in cat_list]
            time_idxs = [time_to_bin(t, self.time_bins) for t in time_list]

            for end in range(self.min_seq_len, len(poi_idxs)):
                start = max(0, end - self.max_seq_len)
                seq_pois = poi_idxs[start:end]
                seq_cats = cat_idxs[start:end]
                seq_times = time_idxs[start:end]
                target_poi = poi_idxs[end]
                target_cat = cat_idxs[end]
                target_time = time_idxs[end]
                if target_poi == PAD_TOKEN or target_cat == PAD_TOKEN:
                    continue
                if not any(poi != PAD_TOKEN for poi in seq_pois):
                    continue
                summary_text = build_history_prompt(
                    poi_ids=poi_list[start:end],
                    categories=cat_list[start:end],
                    times=time_list[start:end],
                )
                flow_feats = self._build_flow_features(seq_pois, existing_maps)
                intent_feat = build_intent_features(
                    poi_idxs=seq_pois,
                    cat_idxs=seq_cats,
                    time_idxs=seq_times,
                    time_bins=self.time_bins,
                    max_seq_len=self.max_seq_len,
                )
                samples.append(
                    {
                        "poi_seq": seq_pois,
                        "cat_seq": seq_cats,
                        "time_seq": seq_times,
                        "flow_feat": flow_feats,
                        "intent_feat": intent_feat,
                        "history_text": summary_text,
                        "target_poi": target_poi,
                        "target_cat": target_cat,
                        "target_time": target_time,
                    }
                )
                if self.max_samples is not None and len(samples) >= self.max_samples:
                    return samples

        return samples

    def _build_flow_features(self, seq_pois: List[int], existing_maps: Optional[Dict]) -> List[List[float]]:
        if existing_maps is None and self.split != "train":
            return [[0.0, 0.0] for _ in seq_pois]

        features = []
        for poi in seq_pois:
            outgoing, incoming = self.flow_map.get(poi, (0.0, 0.0))
            features.append([float(outgoing), float(incoming)])
        return features

    def _build_flow_map(self):
        return self._build_flow_map_from_data()

    def _build_flow_map_from_data(self):
        outgoing_counts = Counter()
        incoming_counts = Counter()
        group_col = self.trajectory_col or self.user_col
        groups = self.data.groupby(group_col)
        for _, group in groups:
            poi_list = group[self.poi_col].astype(str).tolist()
            poi_idxs = [self.poi2idx.get(p, PAD_TOKEN) for p in poi_list]
            for src, dst in zip(poi_idxs, poi_idxs[1:]):
                outgoing_counts[src] += 1
                incoming_counts[dst] += 1

        flow_map = {}
        for poi in self.poi2idx.values():
            flow_map[poi] = (
                outgoing_counts[poi],
                incoming_counts[poi],
            )
        return flow_map

    @property
    def num_pois(self) -> int:
        return len(self.poi2idx)

    @property
    def num_categories(self) -> int:
        return len(self.cat2idx)

    @property
    def num_time_bins(self) -> int:
        return self.time_bins


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    poi_seqs = [item["poi_seq"] for item in batch]
    cat_seqs = [item["cat_seq"] for item in batch]
    time_seqs = [item["time_seq"] for item in batch]
    flow_feats = [item["flow_feat"] for item in batch]
    intent_feats = [item["intent_feat"] for item in batch]
    padded_pois = pad_sequence(poi_seqs, batch_first=True, padding_value=PAD_TOKEN)
    padded_cats = pad_sequence(cat_seqs, batch_first=True, padding_value=PAD_TOKEN)
    padded_times = pad_sequence(time_seqs, batch_first=True, padding_value=PAD_TOKEN)
    padded_flow = pad_sequence(flow_feats, batch_first=True, padding_value=0.0)
    attention_mask = (padded_pois != PAD_TOKEN).long()
    return {
        "poi_seq": padded_pois,
        "cat_seq": padded_cats,
        "time_seq": padded_times,
        "flow_feat": padded_flow,
        "intent_feat": torch.stack(intent_feats),
        "attention_mask": attention_mask,
        "history_text": [item["history_text"] for item in batch],
        "target_poi": torch.stack([item["target_poi"] for item in batch]),
        "target_cat": torch.stack([item["target_cat"] for item in batch]),
        "target_time": torch.stack([item["target_time"] for item in batch]),
    }
