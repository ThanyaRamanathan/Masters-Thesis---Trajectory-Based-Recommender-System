#!/usr/bin/env python
"""Integration tests - validates pipeline without requiring full model load on CPU"""
import torch
import sys
from src.data import TrajectoryDataset
from src.model import GETNextFlowModel
from src.config import TrainingConfig
from torch.utils.data import DataLoader

print("=" * 60)
print("INTEGRATION TEST SUITE - GETNext + text summarisation")
print("=" * 60)

# ========== TEST 1: Data Pipeline ==========
print("\n[1/4] Testing Data Pipeline...")
try:
    dataset = TrajectoryDataset('dataset/NYC/NYC_train.csv')
    print(f"  ✓ Dataset loaded: {len(dataset)} sequences")
    
    if len(dataset) > 0:
        batch = dataset[0]
        print(f"  ✓ Sample batch keys: {list(batch.keys())}")
        print(f"  ✓ History text length: {len(batch['history_text'])} chars")
        print(f"  ✓ Target POI: {batch['target_poi']}, Category: {batch['target_cat']}, Time bin: {batch['target_time']}")
    else:
        print("  ✗ Dataset is empty!")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {str(e)[:100]}")
    sys.exit(1)

# ========== TEST 2: GETNextFlowModel Architecture ==========
print("\n[2/4] Testing GETNextFlowModel Architecture...")
try:
    config = TrainingConfig()
    model = GETNextFlowModel(
        num_pois=10,
        num_categories=5,
        num_time_bins=config.time_bins,
        config=config,
    )
    print(f"  ✓ Model created")
    
    # Test forward pass
    batch_size, max_seq_len = 2, config.max_seq_len
    poi_seq = torch.randint(0, 10, (batch_size, max_seq_len))
    cat_seq = torch.randint(0, 5, (batch_size, max_seq_len))
    time_seq = torch.randint(0, config.time_bins, (batch_size, max_seq_len))
    flow_feat = torch.randn(batch_size, max_seq_len, 2)
    summary_embed = torch.randn(batch_size, config.summary_embed_dim)
    attention_mask = torch.ones(batch_size, max_seq_len)
    
    out_poi, out_cat, out_time = model(poi_seq, cat_seq, time_seq, flow_feat, summary_embed, attention_mask)
    
    assert out_poi.shape == (batch_size, 10), f"POI shape wrong: {out_poi.shape}"
    assert out_cat.shape == (batch_size, 5), f"CAT shape wrong: {out_cat.shape}"
    assert out_time.shape == (batch_size, config.time_bins), f"TIME shape wrong: {out_time.shape}"
    print(f"  ✓ Forward pass OK (POI: {out_poi.shape}, CAT: {out_cat.shape}, TIME: {out_time.shape})")
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {str(e)[:100]}")
    sys.exit(1)

# ========== TEST 3: DataLoader Batching ==========
print("\n[3/4] Testing DataLoader Batching...")
try:
    from src.data import collate_batch
    
    dataset = TrajectoryDataset('dataset/NYC/NYC_train.csv')
    if len(dataset) >= 2:
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_batch)
        batch = next(iter(dataloader))
        
        print(f"  ✓ Batch keys: {list(batch.keys())}")
        print(f"  ✓ POI seq shape: {batch['poi_seq'].shape}")
        print(f"  ✓ Cat seq shape: {batch['cat_seq'].shape}")
        print(f"  ✓ Time seq shape: {batch['time_seq'].shape}")
        print(f"  ✓ Flow feat shape: {batch['flow_feat'].shape}")
        print(f"  ✓ Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  ✓ History text: {type(batch['history_text'])} (list of {len(batch['history_text'])} strings)")
        print(f"  ✓ Target POI shape: {batch['target_poi'].shape}")
    else:
        print(f"  ⚠ Skipped (dataset has < 2 samples): {len(dataset)} samples")
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {str(e)[:100]}")
    sys.exit(1)

# ========== TEST 4: TextSummariser Import ==========
print("\n[4/4] Testing TextSummariser Structure...")
try:
    from src.model import TextSummariser
    print(f"  ✓ TextSummariser can be imported")
    
    # Check methods exist
    methods = ['generate_summary', 'forward']
    for method in methods:
        if hasattr(TextSummariser, method):
            print(f"  ✓ Method '{method}' exists")
        else:
            print(f"  ✗ Method '{method}' NOT FOUND")
            sys.exit(1)
    
    print(f"\n  Note: Full model initialization deferred (14GB on CPU)")
    print(f"  GPU recommended for training with text summarisation enabled")
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {str(e)[:100]}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL INTEGRATION TESTS PASSED")
print("=" * 60)
print("\nProject is ready for GitHub deployment!")
print("  - Data pipeline: ✓ Verified")
print("  - Model architecture: ✓ Verified")
print("  - Text summarisation integration: ✓ Code structure validated")
print("\nTo train with text summarisation enabled, use GPU or reduce batch_size for CPU.")
