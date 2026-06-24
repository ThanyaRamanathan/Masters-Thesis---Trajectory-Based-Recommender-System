import argparse
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .data import TrajectoryDataset, collate_batch
from .model import TextSummariser, GETNextFlowModel, GETNextModelWrapper
from .utils import save_metadata, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train GETNext + text summarisation trajectory model")
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="runs/experiment")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--no-summariser",
        action="store_true",
        help="Disable text summariser and run the base GETNext model.",
    )
    return parser.parse_args()


def build_datasets(args, config):
    train_ds = TrajectoryDataset(args.train_file, max_seq_len=config.max_seq_len, time_bins=config.time_bins, split="train")
    val_ds = TrajectoryDataset(args.val_file, max_seq_len=config.max_seq_len, time_bins=config.time_bins, split="val",
                               existing_maps={
                                   "poi2idx": train_ds.poi2idx,
                                   "cat2idx": train_ds.cat2idx,
                                   "idx2poi": train_ds.idx2poi,
                                   "idx2cat": train_ds.idx2cat,
                                   "flow_map": train_ds.flow_map,
                               })
    test_ds = TrajectoryDataset(args.test_file, max_seq_len=config.max_seq_len, time_bins=config.time_bins, split="test",
                                existing_maps={
                                    "poi2idx": train_ds.poi2idx,
                                    "cat2idx": train_ds.cat2idx,
                                    "idx2poi": train_ds.idx2poi,
                                    "idx2cat": train_ds.idx2cat,
                                    "flow_map": train_ds.flow_map,
                                })
    return train_ds, val_ds, test_ds


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            for key in ["poi_seq", "cat_seq", "time_seq", "flow_feat", "attention_mask", "target_poi", "target_cat", "target_time"]:
                batch[key] = batch[key].to(device)
            preds = model(
                batch["poi_seq"],
                batch["cat_seq"],
                batch["time_seq"],
                batch["flow_feat"],
                batch["history_text"],
                batch["attention_mask"],
            )
            loss = sum(loss_fn(pred, batch[target]) for pred, target in zip(preds, ["target_poi", "target_cat", "target_time"]))
            total_loss += loss.item() * batch["poi_seq"].size(0)
            count += batch["poi_seq"].size(0)
    return total_loss / max(count, 1)


def train(args):
    set_seed(args.seed)

    config = TrainingConfig()
    config.output_dir = args.output_dir
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.weight_decay = args.weight_decay
    config.device = args.device

    os.makedirs(config.output_dir, exist_ok=True)
    train_ds, val_ds, test_ds = build_datasets(args, config)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)

    summariser = None if args.no_summariser else TextSummariser(embedding_dim=config.summary_embed_dim)
    model = GETNextFlowModel(
        num_pois=train_ds.num_pois,
        num_categories=train_ds.num_categories,
        num_time_bins=train_ds.num_time_bins,
        config=config,
    )
    model_wrapper = GETNextModelWrapper(model, summariser)
    model_wrapper.to(config.device)

    optimizer = AdamW(model_wrapper.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    for epoch in range(1, config.epochs + 1):
        model_wrapper.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}"):
            for key in ["poi_seq", "cat_seq", "time_seq", "flow_feat", "attention_mask", "target_poi", "target_cat", "target_time"]:
                batch[key] = batch[key].to(config.device)
            optimizer.zero_grad()
            preds = model_wrapper(
                batch["poi_seq"],
                batch["cat_seq"],
                batch["time_seq"],
                batch["flow_feat"],
                batch["history_text"],
                batch["attention_mask"],
            )
            loss = sum(loss_fn(pred, batch[target]) for pred, target in zip(preds, ["target_poi", "target_cat", "target_time"]))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["poi_seq"].size(0)

        train_loss = running_loss / len(train_ds)
        val_loss = evaluate(model_wrapper, val_loader, config.device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_path = os.path.join(config.output_dir, "best_model.pt")
            torch.save(model_wrapper.state_dict(), save_path)
            save_metadata(
                os.path.join(config.output_dir, "metadata.pkl"),
                {
                    "poi2idx": train_ds.poi2idx,
                    "cat2idx": train_ds.cat2idx,
                    "idx2poi": train_ds.idx2poi,
                        "idx2cat": train_ds.idx2cat,
                        "flow_map": train_ds.flow_map,
                    "config": config,
                },
            )
    print("Training complete.")


if __name__ == "__main__":
    train(parse_args())
