#!/usr/bin/env python
"""Test GETNextFlowModel without text summarisation (architecture validation)"""
import torch
from src.model import GETNextFlowModel
from src.config import TrainingConfig

config = TrainingConfig()
print("Testing GETNextFlowModel architecture...")

# Create model
model = GETNextFlowModel(
    num_pois=10,  # Small for testing
    num_categories=5,   # Small for testing
    num_time_bins=config.time_bins,
    config=config,
)
print(f"✓ GETNextFlowModel created")

# Test forward pass with sample batch
batch_size = 4
max_seq_len = config.max_seq_len
num_poi = 10
num_cat = 5

# Create sample inputs
poi_seq = torch.randint(0, num_poi, (batch_size, max_seq_len))
cat_seq = torch.randint(0, num_cat, (batch_size, max_seq_len))
time_seq = torch.randint(0, config.time_bins, (batch_size, max_seq_len))
flow_feat = torch.randn(batch_size, max_seq_len, 2)  # 2-dim flow features
summary_embed = torch.randn(batch_size, config.summary_embed_dim)
attention_mask = torch.ones(batch_size, max_seq_len)

# Forward pass
output_poi, output_cat, output_time = model(
    poi_seq, cat_seq, time_seq, flow_feat, summary_embed, attention_mask
)

print(f"✓ Forward pass successful")
print(f"  Output POI shape: {output_poi.shape} (expected: [{batch_size}, {num_poi}])")
print(f"  Output CAT shape: {output_cat.shape} (expected: [{batch_size}, {num_cat}])")
print(f"  Output TIME shape: {output_time.shape} (expected: [{batch_size}, {config.time_bins}])")

# Verify shapes
assert output_poi.shape == (batch_size, num_poi), f"POI shape mismatch"
assert output_cat.shape == (batch_size, num_cat), f"CAT shape mismatch"
assert output_time.shape == (batch_size, config.time_bins), f"TIME shape mismatch"
print(f"✓ All output shapes correct!")
