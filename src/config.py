from dataclasses import dataclass

@dataclass
class TrainingConfig:
    train_file: str = "dataset/NYC/NYC_train.csv"
    val_file: str = "dataset/NYC/NYC_val.csv"
    test_file: str = "dataset/NYC/NYC_test.csv"
    max_seq_len: int = 32
    min_seq_len: int = 2
    time_bins: int = 48
    poi_embed_dim: int = 128
    cat_embed_dim: int = 32
    time_embed_dim: int = 32
    hidden_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ffn_dim: int = 256
    summary_embed_dim: int = 384
    batch_size: int = 32
    epochs: int = 50
    lr: float = 0.001
    weight_decay: float = 1e-5
    flow_weight: float = 1.0
    output_dir: str = "runs/experiment"
    device: str = "cuda" if False else "cpu"
