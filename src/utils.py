import os
import pickle
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_metadata(path: str, metadata: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(metadata, f)


def load_metadata(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_history_prompt(poi_ids: List[str], categories: List[str], times: List[datetime], max_events: int = 8) -> str:
    events = []
    start = max(0, len(poi_ids) - max_events)
    for poi, cat, time in zip(poi_ids[start:], categories[start:], times[start:]):
        events.append(f"POI {poi} at {time.strftime('%H:%M')} (category {cat})")
    if not events:
        return "The user has no recent trajectory history."
    prefix = "Summarize the user's recent trajectory history before recommendation. "
    return prefix + " Then recommend the next point of interest. History: " + "; ".join(events) + "."


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.fromtimestamp(float(value))
