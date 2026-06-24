import argparse

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
    return parser.parse_args()


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct_poi = 0
    total_correct_cat = 0
    total_correct_time = 0
    total_examples = 0
    loss_fn = torch.nn.CrossEntropyLoss()

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
            total_examples += batch["poi_seq"].size(0)

            poi_preds = preds[0].argmax(dim=-1)
            cat_preds = preds[1].argmax(dim=-1)
            time_preds = preds[2].argmax(dim=-1)
            total_correct_poi += (poi_preds == batch["target_poi"]).sum().item()
            total_correct_cat += (cat_preds == batch["target_cat"]).sum().item()
            total_correct_time += (time_preds == batch["target_time"]).sum().item()

    return {
        "loss": total_loss / max(total_examples, 1),
        "poi_acc": total_correct_poi / max(total_examples, 1),
        "cat_acc": total_correct_cat / max(total_examples, 1),
        "time_acc": total_correct_time / max(total_examples, 1),
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
    model_wrapper = GETNextModelWrapper(model, summariser)
    model_wrapper.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model_wrapper.to(args.device)

    metrics = evaluate(model_wrapper, test_loader, args.device)
    print("Evaluation results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main(parse_args())
