import argparse
from datetime import datetime
from typing import List

import torch
from .model import TextSummariser, GETNextFlowModel, GETNextModelWrapper
from .utils import load_metadata, build_history_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for GETNext + text summarisation model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--history", type=str, required=True,
                        help="Comma-separated sequence of POI ids")
    parser.add_argument("--categories", type=str, required=True,
                        help="Comma-separated categories for each history POI")
    parser.add_argument("--times", type=str, required=True,
                        help="Comma-separated ISO datetimes for each history POI")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--no-summariser",
        action="store_true",
        help="Disable text summariser and perform inference with the base GETNext model.",
    )
    return parser.parse_args()


def parse_history(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_datetimes(text: str):
    return [datetime.fromisoformat(item.strip()) for item in text.split(",") if item.strip()]


def main(args):
    metadata = load_metadata(args.meta)

    poi2idx = metadata["poi2idx"]
    cat2idx = metadata["cat2idx"]
    idx2poi = metadata["idx2poi"]
    idx2cat = metadata["idx2cat"]
    config = metadata["config"]

    history_pois = parse_history(args.history)
    history_cats = parse_history(args.categories)
    history_times = parse_datetimes(args.times)

    if len(history_pois) != len(history_cats) or len(history_pois) != len(history_times):
        raise ValueError("History POIs, categories, and times must have the same length")

    summary = build_history_prompt(history_pois, history_cats, history_times, max_events=config.max_seq_len)

    poi_seq = torch.tensor([[poi2idx.get(p, 0) for p in history_pois]], dtype=torch.long)
    cat_seq = torch.tensor([[cat2idx.get(c, 0) for c in history_cats]], dtype=torch.long)
    time_seq = torch.tensor([[int((t.hour * 3600 + t.minute * 60 + t.second) / (24 * 3600 / config.time_bins)) for t in history_times]], dtype=torch.long)
    flow_feat = torch.zeros((1, poi_seq.size(1), 2), dtype=torch.float)
    attention_mask = (poi_seq != 0).long()

    model = GETNextFlowModel(
        num_pois=len(poi2idx),
        num_categories=len(cat2idx),
        num_time_bins=config.time_bins,
        config=config,
    )
    summariser = None if args.no_summariser else TextSummariser(embedding_dim=config.summary_embed_dim)
    wrapped = GETNextModelWrapper(model, summariser)
    wrapped.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    wrapped.to(args.device)
    wrapped.eval()

    poi_seq = poi_seq.to(args.device)
    cat_seq = cat_seq.to(args.device)
    time_seq = time_seq.to(args.device)
    flow_feat = flow_feat.to(args.device)
    attention_mask = attention_mask.to(args.device)

    with torch.no_grad():
        poi_logits, cat_logits, time_logits = wrapped(poi_seq, cat_seq, time_seq, flow_feat, [summary], attention_mask)
    next_poi = idx2poi[int(poi_logits.argmax(dim=-1).item())]
    next_cat = idx2cat[int(cat_logits.argmax(dim=-1).item())]
    next_time_bin = int(time_logits.argmax(dim=-1).item())
    print("Next POI:", next_poi)
    print("Next category:", next_cat)
    print("Next time bin:", next_time_bin)


if __name__ == "__main__":
    main(parse_args())
