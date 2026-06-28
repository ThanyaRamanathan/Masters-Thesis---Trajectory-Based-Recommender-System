import argparse
import json
import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .config import TrainingConfig
from .data import INTENT_FEATURE_NAMES, TrajectoryDataset, collate_batch
from .evaluate import evaluate
from .model import GETNextFlowModel, GETNextModelWrapper
from .utils import save_metadata, set_seed


class MiniLMHistorySummariser(nn.Module):
    """Lightweight history-text encoder used for a runnable local ablation."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", freeze=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, texts):
        device = next(self.backbone.parameters()).device
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.backbone(**tokens)
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        masked = outputs.last_hidden_state * attention_mask
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        return masked.sum(dim=1) / lengths


def parse_args():
    parser = argparse.ArgumentParser(description="Fairly compare GETNext baseline and summariser variants")
    parser.add_argument("--train-file", type=str, default="dataset/NYC/NYC_train.csv")
    parser.add_argument("--val-file", type=str, default="dataset/NYC/NYC_val.csv")
    parser.add_argument("--test-file", type=str, default="dataset/NYC/NYC_test.csv")
    parser.add_argument("--output-dir", type=str, default="runs/fair_compare")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline,summariser,intent,summariser_intent",
        help="Comma-separated variants: baseline,summariser,intent,summariser_intent",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--train-summariser",
        action="store_true",
        help="Fine-tune MiniLM during the summariser run. By default it is frozen for speed/reproducibility.",
    )
    return parser.parse_args()


def build_datasets(args, config):
    train_ds = TrajectoryDataset(
        args.train_file,
        max_seq_len=config.max_seq_len,
        time_bins=config.time_bins,
        split="train",
        max_samples=args.max_train_samples,
    )
    maps = {
        "poi2idx": train_ds.poi2idx,
        "cat2idx": train_ds.cat2idx,
        "idx2poi": train_ds.idx2poi,
        "idx2cat": train_ds.idx2cat,
        "flow_map": train_ds.flow_map,
    }
    val_ds = TrajectoryDataset(
        args.val_file,
        max_seq_len=config.max_seq_len,
        time_bins=config.time_bins,
        split="val",
        existing_maps=maps,
        max_samples=args.max_val_samples,
    )
    test_ds = TrajectoryDataset(
        args.test_file,
        max_seq_len=config.max_seq_len,
        time_bins=config.time_bins,
        split="test",
        existing_maps=maps,
        max_samples=args.max_test_samples,
    )
    return train_ds, val_ds, test_ds


def make_config(args, output_dir):
    config = TrainingConfig()
    config.train_file = args.train_file
    config.val_file = args.val_file
    config.test_file = args.test_file
    config.output_dir = output_dir
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.weight_decay = args.weight_decay
    config.device = args.device
    return config


def train_variant(args, variant, use_summariser, use_intent):
    set_seed(args.seed)
    run_dir = os.path.join(args.output_dir, variant)
    os.makedirs(run_dir, exist_ok=True)

    config = make_config(args, run_dir)
    train_ds, val_ds, test_ds = build_datasets(args, config)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)

    model = GETNextFlowModel(
        num_pois=train_ds.num_pois,
        num_categories=train_ds.num_categories,
        num_time_bins=train_ds.num_time_bins,
        config=config,
    )
    summariser = MiniLMHistorySummariser(freeze=not args.train_summariser) if use_summariser else None
    wrapped = GETNextModelWrapper(model, summariser, use_intent=use_intent).to(config.device)

    optimizer = AdamW(
        [param for param in wrapped.parameters() if param.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    history = []
    for epoch in range(1, config.epochs + 1):
        wrapped.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"{variant} {epoch}/{config.epochs}"):
            for key in ["poi_seq", "cat_seq", "time_seq", "flow_feat", "intent_feat", "attention_mask", "target_poi", "target_cat", "target_time"]:
                batch[key] = batch[key].to(config.device)
            optimizer.zero_grad()
            preds = wrapped(
                batch["poi_seq"],
                batch["cat_seq"],
                batch["time_seq"],
                batch["flow_feat"],
                batch["history_text"],
                batch["attention_mask"],
                batch["intent_feat"],
            )
            loss = sum(
                loss_fn(pred, batch[target])
                for pred, target in zip(preds, ["target_poi", "target_cat", "target_time"])
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch["poi_seq"].size(0)

        train_loss = total_loss / max(len(train_ds), 1)
        val_metrics = evaluate(wrapped, val_loader, config.device)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"{variant} epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = deepcopy(wrapped.state_dict())

    if best_state is not None:
        wrapped.load_state_dict(best_state)

    checkpoint_path = os.path.join(run_dir, "best_model.pt")
    torch.save(wrapped.state_dict(), checkpoint_path)
    save_metadata(
        os.path.join(run_dir, "metadata.pkl"),
        {
            "poi2idx": train_ds.poi2idx,
            "cat2idx": train_ds.cat2idx,
            "idx2poi": train_ds.idx2poi,
            "idx2cat": train_ds.idx2cat,
            "flow_map": train_ds.flow_map,
            "config": config,
            "variant": variant,
            "summariser": "minilm-history" if use_summariser else "learned-embedding-baseline",
            "summariser_trainable": bool(use_summariser and args.train_summariser),
            "use_intent": use_intent,
            "intent_feature_names": INTENT_FEATURE_NAMES,
        },
    )

    metrics = {
        "samples": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "best_val_loss": best_val,
        "val": evaluate(wrapped, val_loader, config.device),
        "test": evaluate(wrapped, test_loader, config.device),
        "history": history,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    variant_settings = {
        "baseline": {"use_summariser": False, "use_intent": False},
        "summariser": {"use_summariser": True, "use_intent": False},
        "intent": {"use_summariser": False, "use_intent": True},
        "summariser_intent": {"use_summariser": True, "use_intent": True},
    }
    requested_variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    unknown = [variant for variant in requested_variants if variant not in variant_settings]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid variants: {list(variant_settings)}")

    results = {
        variant: train_variant(args, variant, **variant_settings[variant])
        for variant in requested_variants
    }

    comparison = {
        "settings": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "device": args.device,
            "summariser": "MiniLM history encoder",
            "summariser_trainable": args.train_summariser,
            "intent_feature_names": INTENT_FEATURE_NAMES,
            "variants": requested_variants,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
        },
        **results,
        "delta_vs_baseline": {
            name: {
                key: metrics["test"][key] - results["baseline"]["test"][key]
                for key in metrics["test"]
            }
            for name, metrics in results.items()
            if name != "baseline" and "baseline" in results
        },
    }
    report_path = os.path.join(args.output_dir, "comparison.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\nFair comparison complete")
    print(f"Report: {report_path}")
    print("Test metrics:")
    for name, result in results.items():
        metrics = result["test"]
        print(
            f"{name}: loss={metrics['loss']:.4f}, "
            f"poi_acc={metrics['poi_acc']:.4f}, "
            f"recall@5={metrics['recall@5']:.4f}, "
            f"recall@10={metrics['recall@10']:.4f}, "
            f"ndcg@10={metrics['ndcg@10']:.4f}, "
            f"mrr={metrics['mrr']:.4f}, "
            f"cat_acc={metrics['cat_acc']:.4f}, "
            f"time_acc={metrics['time_acc']:.4f}"
        )


if __name__ == "__main__":
    main(parse_args())
