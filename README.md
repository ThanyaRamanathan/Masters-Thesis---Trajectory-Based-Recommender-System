# Thesis Rebuild: GETNext + Text Summarisation

A reconstructed Masters thesis project based on the GETNext trajectory flow model, extended with a text summarisation prompt before the recommendation layer.

## Project Overview

This project uses:
- Python + PyTorch
- Foursquare NYC trajectory data
- A reconstructed GETNext-style flow-enhanced transformer
- Text summarisation of user history as an auxiliary feature
- A next POI, time, and category prediction head

## Model Explanation

The core idea is to predict the next point of interest (POI), its category, and the likely time bin from a user's recent trajectory history.

### 1. Sequence encoder
The model consumes a sequence of recent POIs, categories, and time bins. Each input is embedded separately and then combined with a lightweight flow feature that captures how often a POI tends to be followed by another POI in the training data.

### 2. Transformer backbone
The concatenated embeddings are passed through a Transformer encoder. This allows the model to learn contextual dependencies across the user's recent visits rather than treating each step independently.

### 3. Prediction heads
From the final hidden state of the sequence, three output heads predict:
- the next POI,
- the next category,
- the next time bin.

The training objective is the sum of three cross-entropy losses, one per prediction head.

### 4. Text summariser branch
To enrich the sequence representation, the model can optionally condition on a text summary of the recent trajectory history. The summariser converts the history text into an embedding that is injected into the sequence representation before the Transformer layer. This provides a semantic view of the user's behaviour beyond raw IDs and timestamps.

### 5. Baseline ablation
The repository also supports a no-summariser baseline. In that setting, the model uses a learned summary embedding instead of the text summariser output, allowing direct comparison between the base GETNext-style model and the summariser-augmented version.

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
- The code is designed so the text summariser can be enabled or disabled easily, which makes ablation studies straightforward.
