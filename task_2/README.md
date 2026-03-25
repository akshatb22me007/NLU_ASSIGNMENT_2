# Task 2: Character-Level Name Generation (RNN, BLSTM, Attention RNN)

This task trains three character-level sequence models to generate Indian names:
- Vanilla RNN
- BLSTM
- Attention RNN

Recommended run order:
1. Run `generate_name.py` to create `TrainingNames.txt` using Gemini.
2. Run `main.py` to train all models, generate names, evaluate metrics, and save plots.

## Folder Contents

- `generate_name.py`: Calls Gemini API to create a list of names and saves them to `TrainingNames.txt`.
- `TrainingNames.txt`: Training corpus used by all models (one name per line).
- `data_utils.py`: Data loading, vocabulary building, encoding/decoding, dataset class, and padding collate function.
- `models.py`: Implements `VanillaRNN`, `BLSTM` (custom LSTM cells), and `AttentionRNN`.
- `train.py`: Shared training loop (cross-entropy loss + Adam optimizer).
- `inference.py`: Sampling utilities to generate names from a trained model.
- `eval.py`: Quality metrics for generated names (`novelty`, `diversity`).
- `main.py`: End-to-end pipeline for training all 3 models and saving outputs.
- `models/`: Saved model checkpoints (`.pt`).
- `outputs/`: Generated names and visualization plots.

## Prerequisites

- Python 3.9+
- PyTorch
- Matplotlib (optional, for plots)
- Google Gemini client library (`google-generativeai`) for data generation

Install dependencies:

```bash
pip install torch matplotlib google-generativeai
```

## Step 1: Generate Training Data

Run from this folder:

```bash
cd task_2
python generate_name.py
```

What it does:
- Queries Gemini in batches.
- Collects names until around 1000 unique names.
- Saves final list in `TrainingNames.txt`.

Important:
- `generate_name.py` Use your Gemini API key in `genai.configure(api_key=...)` call.

## Step 2: Train Models and Generate Outputs

After `TrainingNames.txt` is ready, run:

```bash
python main.py
```

What `main.py` does:
1. Loads names from `TrainingNames.txt`.
2. Builds character vocabulary (including start/end tokens and padding).
3. Trains the three models sequentially:
   - `vanilla_rnn` for 30 epochs
   - `blstm` for 25 epochs
   - `attention_rnn` for 60 epochs
4. Saves checkpoints in `models/`.
5. Generates 200 names per model and saves to `outputs/`.
6. Computes novelty/diversity scores per model.
7. Saves plots for loss curves, metric comparison, and name-length distributions.

## Expected Artifacts

After a successful run:

- Model checkpoints:
  - `models/vanilla_rnn.pt`
  - `models/blstm.pt`
  - `models/attention_rnn.pt`

- Generated names:
  - `outputs/generated_names_vanilla_rnn.txt`
  - `outputs/generated_names_blstm.txt`
  - `outputs/generated_names_attention_rnn.txt`

- Plots:
  - `outputs/training_loss_curves.png`
  - `outputs/metrics_comparison.png`
  - `outputs/name_length_distribution_vanilla_rnn.png`
  - `outputs/name_length_distribution_blstm.png`
  - `outputs/name_length_distribution_attention_rnn.png`

## Scripts

### `generate_name.py`
- Uses Gemini with a fixed prompt to request Indian names.
- Cleans response lines (`isalpha`, lowercase, min length > 2).
- Accumulates unique names and writes `TrainingNames.txt`.

### `data_utils.py`
- `load_names`: Reads names and wraps each as `<name>` for sequence boundaries.
- `build_vocab`: Builds char vocabulary and optional pad token (`~`).
- `NameDataset`: Produces `(input_seq, target_seq)` by next-character shift.
- `build_collate_fn`: Pads variable-length batches.

### `models.py`
- `VanillaRNN`: Manual recurrent update with tanh.
- `BLSTM`: Bidirectional custom LSTM cell stack; concatenates forward/backward states.
- `AttentionRNN`: RNN with attention over previously seen hidden states at each step.

### `train.py`
- Shared trainer for any model returning logits `[B, T, V]`.
- Uses `CrossEntropyLoss(ignore_index=-100)` so padded targets do not affect loss.

### `inference.py`
- Autoregressive character sampling starting from `<` until `>` or max length.
- Applies softmax + multinomial sampling to pick next character.

### `eval.py`
- `novelty`: fraction of generated names not present in training set.
- `diversity`: fraction of unique names in generated outputs.

### `main.py`
- Orchestrates full experiment and writes all checkpoints, text outputs, and figures.
