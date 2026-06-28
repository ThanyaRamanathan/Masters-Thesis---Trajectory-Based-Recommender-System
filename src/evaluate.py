import argparse
import math

import torch
from torch.utils.data import DataLoader

from .data import TrajectoryDataset, collate_batch
from .model import TextSummariser, GETNextFlowModel, GETNextModelWrapper
from .utils import load_metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the GETNext + text summarisation model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--no-summariser",
        action="store_true",
        help="Disable text summariser and evaluate the base GETNext model.",
    )
    parser.add_argument(
        "--use-intent",
        action="store_true",
        help="Fuse structured trajectory intent features into the GETNext encoder.",
    )
    return parser.parse_args()


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct_poi = 0
    total_correct_cat = 0
    total_correct_time = 0
    total_examples = 0
    recall_hits = {5: 0, 10: 0, 20: 0}
    ndcg_scores = {5: 0.0, 10: 0.0, 20: 0.0}
    reciprocal_rank_total = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            for key in ["poi_seq", "cat_seq", "time_seq", "flow_feat", "intent_feat", "attention_mask", "target_poi", "target_cat", "target_time"]:
                batch[key] = batch[key].to(device)
            preds = model(
                batch["poi_seq"],
                batch["cat_seq"],
                batch["time_seq"],
                batch["flow_feat"],
                batch["history_text"],
                batch["attention_mask"],
                batch["intent_feat"],
            )
            loss = sum(loss_fn(pred, batch[target]) for pred, target in zip(preds, ["target_poi", "target_cat", "target_time"]))
            total_loss += loss.item() * batch["poi_seq"].size(0)
            total_examples += batch["poi_seq"].size(0)

            poi_preds = preds[0].argmax(dim=-1)
            cat_preds = preds[1].argmax(dim=-1)
            time_preds = preds[2].argmax(dim=-1)
            total_correct_poi += (poi_preds == batch["target_poi"]).sum().item()
            total_correct_cat += (cat_preds == batch["target_cat"]).sum().item()
            total_correct_time += (time_preds == batch["target_time"]).sum().item()

            poi_logits = preds[0]
            target_pois = batch["target_poi"]
            max_k = min(20, poi_logits.size(-1))
            top_indices = torch.topk(poi_logits, k=max_k, dim=-1).indices
            matches = top_indices == target_pois.unsqueeze(-1)
            for row in matches:
                match_positions = torch.nonzero(row, as_tuple=False)
                if match_positions.numel() == 0:
                    continue
                rank = int(match_positions[0].item()) + 1
                reciprocal_rank_total += 1.0 / rank
                for k in recall_hits:
                    if rank <= min(k, max_k):
                        recall_hits[k] += 1
                        ndcg_scores[k] += 1.0 / math.log2(rank + 1)

    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "poi_acc": total_correct_poi / max(total_examples, 1),
        "cat_acc": total_correct_cat / max(total_examples, 1),
        "time_acc": total_correct_time / max(total_examples, 1),
        "mrr": reciprocal_rank_total / max(total_examples, 1),
    }
    for k in recall_hits:
        metrics[f"recall@{k}"] = recall_hits[k] / max(total_examples, 1)
        metrics[f"ndcg@{k}"] = ndcg_scores[k] / max(total_examples, 1)
    return {
        **metrics,
    }


def main(args):
    metadata = load_metadata(args.meta)

    train_maps = {
        "poi2idx": metadata["poi2idx"],
        "cat2idx": metadata["cat2idx"],
        "idx2poi": metadata["idx2poi"],
        "idx2cat": metadata["idx2cat"],
        "flow_map": metadata.get("flow_map"),
    }
    test_ds = TrajectoryDataset(args.test_file, split="test", existing_maps=train_maps)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = GETNextFlowModel(
        num_pois=test_ds.num_pois,
        num_categories=test_ds.num_categories,
        num_time_bins=test_ds.num_time_bins,
        config=metadata["config"],
    )
    summariser = None if args.no_summariser else TextSummariser(embedding_dim=metadata["config"].summary_embed_dim)
    model_wrapper = GETNextModelWrapper(model, summariser, use_intent=args.use_intent or metadata.get("use_intent", False))
    load_result = model_wrapper.load_state_dict(torch.load(args.checkpoint, map_location=args.device), strict=False)
    if load_result.missing_keys:
        print(f"Warning: missing checkpoint keys ignored: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Warning: unexpected checkpoint keys ignored: {load_result.unexpected_keys}")
    model_wrapper.to(args.device)

    metrics = evaluate(model_wrapper, test_loader, args.device)
    print("Evaluation results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main(parse_args())
