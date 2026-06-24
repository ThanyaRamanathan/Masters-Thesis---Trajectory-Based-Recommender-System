#!/usr/bin/env python
"""Test data loading pipeline"""
from src.data import TrajectoryDataset

# Test with real NYC data
dataset = TrajectoryDataset('dataset/NYC/NYC_train.csv')
print(f'✓ Dataset created with {len(dataset)} sequences')

if len(dataset) > 0:
    # Test getting a sample
    batch = dataset[0]
    print(f'✓ Sample batch keys: {list(batch.keys())}')
    print(f'✓ History text: {batch["history_text"][:100]}...')
    print(f'✓ Target POI ID: {batch["target_poi"]}')
    print(f'✓ Target category: {batch["target_cat"]}')
    print(f'✓ Target time bin: {batch["target_time"]}')
else:
    print('✗ Dataset is empty')
