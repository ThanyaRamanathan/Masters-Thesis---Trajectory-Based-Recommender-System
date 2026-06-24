# Thesis Rebuild: GETNext + Text Summarisation

A reconstructed Masters thesis project based on the GETNext trajectory flow model, extended with a text summarisation prompt before the recommendation layer.

## Project Overview

This project uses:
- Python + PyTorch
- Foursquare NYC trajectory data
- A reconstructed GETNext-style flow-enhanced transformer
- Text summarisation of user history as an auxiliary feature
- A next POI, time, and category prediction head

## Repository Structure

- `src/` — model, data pipeline, training, evaluation, inference, and utilities
- `dataset/` — expected location for NYC trajectory CSV files
- `requirements.txt` — Python dependencies
- `README.md` — project documentation
- `LICENSE` — project license

## Installation

1. Create a Python environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Place the Foursquare NYC files under `dataset/NYC/` with names:
- `NYC_train.csv`
- `NYC_val.csv`
- `NYC_test.csv`

The dataset loader expects the following columns:
- `user_id`
- `poi_id`
- `category` or `poi_category`
- `timestamp` or `time`
- optional `latitude`, `longitude`

If your files use different names, rename the columns or adjust the loader in `src/data.py`.

## Training

Train with text summarisation enabled:

```bash
python -m src.train \
  --train-file dataset/NYC/NYC_train.csv \
  --val-file dataset/NYC/NYC_val.csv \
  --test-file dataset/NYC/NYC_test.csv \
  --output-dir runs/exp1
```

Train the base GETNext model without text summarisation:

```bash
python -m src.train \
  --train-file dataset/NYC/NYC_train.csv \
  --val-file dataset/NYC/NYC_val.csv \
  --test-file dataset/NYC/NYC_test.csv \
  --output-dir runs/exp1_baseline \
  --no-summariser
```

Use the same flags for evaluation and inference when comparing baselines.

## Evaluation

```bash
python -m src.evaluate --checkpoint runs/exp1/best_model.pt --meta runs/exp1/metadata.pkl --test-file dataset/NYC/NYC_test.csv
```

## Inference

```bash
python -m src.infer --checkpoint runs/exp1/best_model.pt --meta runs/exp1/metadata.pkl --history "1,23,45" --categories "Food,Park,Museum" --times "2024-05-01T12:00:00,2024-05-01T12:40:00,2024-05-01T13:20:00"
```

## Notes

- The model builds a trajectory flow map from training data, similar to GETNext.
- A text prompt summarises the trajectory history before recommendation.
- The project includes training, evaluation, and inference flows for reproducible experiments.
